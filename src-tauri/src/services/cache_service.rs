//! SQLite-based cache service for storing extracted features.
//!
//! Uses the exact same per-folder database layout as the original .NET app
//! (`FolderDatabaseCacheService`): a `.differ_cache.db` file at the folder
//! root with an `ImageFeatures` table keyed by FilePath and validated by
//! FileSize + LastModified (Windows FILETIME). Caches produced by the .NET
//! version are therefore reused as-is, and vice versa.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use rusqlite::{params, Connection};
use crate::state::CacheStats;

/// Cache database file name (same as the .NET version).
const CACHE_DB_NAME: &str = ".differ_cache.db";

/// Offset between the Windows FILETIME epoch (1601-01-01) and Unix epoch,
/// in 100ns ticks.
const FILETIME_UNIX_DIFF: i64 = 116_444_736_000_000_000;

/// Get the cache database path for a folder.
fn get_cache_path(folder_path: &str) -> String {
    Path::new(folder_path).join(CACHE_DB_NAME).to_string_lossy().to_string()
}

/// Convert a file's mtime to Windows FILETIME (100ns ticks since 1601-01-01),
/// matching .NET `FileInfo.LastWriteTimeUtc.ToFileTimeUtc()`.
fn to_filetime(t: SystemTime) -> i64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => FILETIME_UNIX_DIFF + (d.as_nanos() / 100) as i64,
        Err(e) => FILETIME_UNIX_DIFF - (e.duration().as_nanos() / 100) as i64,
    }
}

/// File identity used as the cache validation key: (size, FILETIME mtime).
fn file_identity(path: &str) -> Result<(i64, i64), String> {
    let meta = std::fs::metadata(path)
        .map_err(|e| format!("Failed to stat {}: {}", path, e))?;
    let modified = meta.modified()
        .map_err(|e| format!("Failed to read mtime of {}: {}", path, e))?;
    Ok((meta.len() as i64, to_filetime(modified)))
}

/// Current unix epoch seconds (for CreatedAt/UpdatedAt defaults).
fn now_epoch() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Open or create a cache database for a folder (schema identical to .NET).
pub fn open_cache(folder_path: &str) -> Result<Connection, String> {
    let db_path = get_cache_path(folder_path);

    // Delete a corrupted database file, like the .NET version does
    if Path::new(&db_path).exists() {
        let ok = Connection::open_with_flags(
            &db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .and_then(|c| {
            c.query_row(
                "SELECT name FROM sqlite_master WHERE type='table' LIMIT 1",
                [],
                |_| Ok(()),
            )
            .or(Ok(()))
        })
        .is_ok();
        if !ok {
            log::warn!("Deleting corrupted cache database: {}", db_path);
            let _ = std::fs::remove_file(&db_path);
        }
    }

    let conn = Connection::open(&db_path)
        .map_err(|e| format!("Failed to open cache database: {}", e))?;

    // WAL + NORMAL sync: feature blobs are large (hundreds of KB per image)
    // and written from a background thread; WAL decouples readers from the
    // writer and removes the per-transaction fsync stall. .NET readers
    // (Microsoft.Data.Sqlite) read WAL databases transparently.
    let _: String = conn
        .query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0))
        .map_err(|e| format!("Failed to set WAL mode: {}", e))?;
    conn.execute_batch("PRAGMA synchronous=NORMAL;")
        .map_err(|e| format!("Failed to set synchronous mode: {}", e))?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS ImageFeatures (
            FilePath TEXT PRIMARY KEY,
            FileName TEXT NOT NULL,
            FileSize INTEGER NOT NULL,
            LastModified INTEGER NOT NULL,
            DinoFeatures BLOB NOT NULL,
            DinoFeatureLength INTEGER NOT NULL,
            CreatedAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
            UpdatedAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_imagefeatures_lastmodified ON ImageFeatures(LastModified);
        CREATE TABLE IF NOT EXISTS DatabaseVersion (
            Version INTEGER PRIMARY KEY,
            AppliedAt INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );
        INSERT OR IGNORE INTO DatabaseVersion (Version) VALUES (1);",
    )
    .map_err(|e| format!("Failed to create cache tables: {}", e))?;

    Ok(conn)
}

/// Retrieve cached features for an image.
///
/// Returns None when not cached, when the file changed (size/mtime mismatch),
/// or when the cached vector's dimension doesn't match `expected_len` (model
/// or input-resolution changed since it was written) — all are re-extracted.
/// The file-identity lookup matches the .NET
/// `WHERE FilePath = ? AND FileSize = ? AND LastModified = ?`.
pub fn get_features(
    conn: &Connection,
    path: &str,
    expected_len: usize,
) -> Result<Option<Vec<f32>>, String> {
    let (file_size, last_modified) = file_identity(path)?;

    let mut stmt = conn
        .prepare_cached(
            "SELECT DinoFeatures, DinoFeatureLength FROM ImageFeatures
             WHERE FilePath = ?1 AND FileSize = ?2 AND LastModified = ?3
             LIMIT 1",
        )
        .map_err(|e| format!("Failed to prepare query: {}", e))?;

    let row = stmt.query_row(params![path, file_size, last_modified], |row| {
        let bytes: Vec<u8> = row.get(0)?;
        let len: i64 = row.get(1)?;
        Ok((bytes, len))
    });

    match row {
        Ok((bytes, len)) => {
            let features: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            if features.is_empty()
                || features.len() != len as usize
                || features.len() != expected_len
            {
                // Inconsistent row or stale dimension: treat as miss so it
                // gets re-extracted with the active model configuration
                return Ok(None);
            }
            Ok(Some(features))
        }
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(format!("Failed to query cache: {}", e)),
    }
}

/// Store a feature vector in the cache (INSERT OR REPLACE, .NET layout).
pub fn store_features(conn: &Connection, path: &str, features: &[f32]) -> Result<(), String> {
    let (file_size, last_modified) = file_identity(path)?;
    let file_name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let feature_bytes: Vec<u8> = features.iter().flat_map(|f| f.to_le_bytes()).collect();

    conn.execute(
        "INSERT OR REPLACE INTO ImageFeatures
         (FilePath, FileName, FileSize, LastModified, DinoFeatures, DinoFeatureLength, CreatedAt, UpdatedAt)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
        params![path, file_name, file_size, last_modified, feature_bytes, features.len() as i64, now_epoch()],
    )
    .map_err(|e| format!("Failed to store features: {}", e))?;

    Ok(())
}

/// Store several feature vectors inside a single transaction.
pub fn store_features_batch(
    conn: &mut Connection,
    items: &[(&str, &Vec<f32>)],
) -> Result<(), String> {
    let tx = conn
        .transaction()
        .map_err(|e| format!("Failed to start transaction: {}", e))?;
    for (path, features) in items {
        store_features(&tx, path, features)?;
    }
    tx.commit().map_err(|e| format!("Failed to commit cache batch: {}", e))
}

/// Get cache statistics for a folder.
pub fn get_cache_stats(folder_path: &str) -> Result<CacheStats, String> {
    let db_path = get_cache_path(folder_path);
    if !Path::new(&db_path).exists() {
        return Ok(CacheStats {
            cached_count: 0,
            cache_size_bytes: 0,
            is_valid: false,
        });
    }

    let conn = open_cache(folder_path)?;
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM ImageFeatures", [], |row| row.get(0))
        .map_err(|e| format!("Failed to count cache entries: {}", e))?;

    let cache_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);

    Ok(CacheStats {
        cached_count: count as usize,
        cache_size_bytes: cache_size,
        is_valid: true,
    })
}

/// Clear the cache for a folder by deleting the database file.
pub fn clear_cache(folder_path: &str) -> Result<(), String> {
    let db_path = get_cache_path(folder_path);
    if Path::new(&db_path).exists() {
        std::fs::remove_file(&db_path)
            .map_err(|e| format!("Failed to delete cache database: {}", e))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cache_roundtrip_and_invalidation() {
        let temp_dir = tempdir().unwrap();
        let folder_path = temp_dir.path().to_str().unwrap();

        // A real file is needed since the cache key derives from file metadata
        let img_path = temp_dir.path().join("test.jpg");
        std::fs::write(&img_path, b"fake image data").unwrap();
        let img_path_str = img_path.to_str().unwrap();

        let mut conn = open_cache(folder_path).unwrap();

        let features = vec![1.0f32, 2.0, 3.0, 4.0];
        store_features(&conn, img_path_str, &features).unwrap();

        let retrieved = get_features(&conn, img_path_str, 4).unwrap().unwrap();
        assert_eq!(retrieved.len(), 4);
        assert!((retrieved[0] - 1.0).abs() < 1e-6);

        // A dimension mismatch (e.g. model/input resolution changed)
        // invalidates the entry
        assert!(get_features(&conn, img_path_str, 8).unwrap().is_none());

        // Modifying the file invalidates the entry (size changes)
        std::fs::write(&img_path, b"changed image data!!").unwrap();
        assert!(get_features(&conn, img_path_str, 4).unwrap().is_none());

        // Batch store
        let f2 = vec![5.0f32, 6.0];
        store_features_batch(&mut conn, &[(img_path_str, &f2)]).unwrap();
        assert_eq!(get_features(&conn, img_path_str, 2).unwrap().unwrap(), f2);

        let stats = get_cache_stats(folder_path).unwrap();
        assert_eq!(stats.cached_count, 1);

        drop(conn);
        clear_cache(folder_path).unwrap();
        assert_eq!(get_cache_stats(folder_path).unwrap().cached_count, 0);
    }
}
