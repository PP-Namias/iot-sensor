# database_manager.py
import sqlite3
import os
import numpy as np
import config # Import configuration
from logging_config import log_event

def init_database():
    """Initializes the SQLite database and creates the necessary tables if they don't exist."""
    try:
        if not os.path.exists(config.IMAGE_DIR):
            os.makedirs(config.IMAGE_DIR)
        conn = sqlite3.connect(config.DB_FILE)
        cursor = conn.cursor()
        # Table for basic driver info
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL
            )
        ''')
        # Table for individual encodings linked to drivers
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS driver_encodings (
                encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_ref_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                encoding BLOB NOT NULL,
                FOREIGN KEY (driver_ref_id) REFERENCES drivers (id) ON DELETE CASCADE
            )
        ''')
        conn.commit()
        print("Database initialized with drivers and driver_encodings tables.")
        log_event("Database initialized successfully")  # Add log entry
    except sqlite3.Error as e:
        print(f"Database initialization error: {e}")
        log_event(f"Database initialization error: {e}", level="error")  # Add log entry
    except OSError as e:
        print(f"Error creating image directory '{config.IMAGE_DIR}': {e}")
        log_event(f"Error creating image directory '{config.IMAGE_DIR}': {e}", level="error")  # Add log entry
    except Exception as e:
        print(f"An unexpected error occurred during database initialization: {e}")
        log_event(f"An unexpected error occurred during database initialization: {e}", level="error")  # Add log entry
    finally:
        if conn:
            conn.close()


def add_driver_db(driver_id, name, image_encoding_pairs):
    """Adds a driver and their associated image paths/encodings to the database."""
    conn = None # Initialize conn to None
    driver_db_id = None
    try:
        conn = sqlite3.connect(config.DB_FILE)
        cursor = conn.cursor()
        # --- Step 1: Add or Find the Driver ---
        cursor.execute("SELECT id FROM drivers WHERE driver_id = ?", (driver_id,))
        result = cursor.fetchone()
        if result:
            driver_db_id = result[0]
            # Optional: Update name if it differs?
            cursor.execute("UPDATE drivers SET name = ? WHERE id = ?", (name, driver_db_id))
            print(f"Driver '{driver_id}' exists (ID: {driver_db_id}). Updating name and adding encodings.")
        else:
            cursor.execute("INSERT INTO drivers (driver_id, name) VALUES (?, ?)", (driver_id, name))
            driver_db_id = cursor.lastrowid
            print(f"Added new driver '{driver_id}' (ID: {driver_db_id}).")

        # --- Step 2: Add Encodings ---
        added_count = 0
        for image_path, encoding in image_encoding_pairs:
            if encoding is None: continue
            try:
                cursor.execute(
                    "INSERT INTO driver_encodings (driver_ref_id, image_path, encoding) VALUES (?, ?, ?)",
                    (driver_db_id, image_path, encoding.tobytes())
                )
                added_count += 1
            except sqlite3.Error as e_inner:
                print(f"Error adding encoding for image {image_path}: {e_inner}")

        conn.commit()
        print(f"Successfully committed {added_count} encodings for driver ID '{driver_id}'.")
        return True

    except sqlite3.IntegrityError:
        print(f"Error: Driver ID '{driver_id}' likely already exists (Integrity constraint violation).")
        if conn: conn.rollback()
        return False
    except sqlite3.Error as e:
        print(f"Database Error adding driver/encodings: {e}")
        if conn: conn.rollback()
        return False
    except Exception as e:
         print(f"Unexpected error in add_driver_db: {e}")
         if conn: conn.rollback()
         return False
    finally:
        if conn:
            conn.close()


def delete_driver_db(driver_id):
    """Deletes a driver and their associated encodings/images."""
    conn = None # Initialize conn
    try:
        conn = sqlite3.connect(config.DB_FILE)
        cursor = conn.cursor()
        # Find driver PK
        cursor.execute("SELECT id FROM drivers WHERE driver_id = ?", (driver_id,))
        driver_result = cursor.fetchone()
        if not driver_result: return False # Driver not found

        driver_db_id = driver_result[0]
        # Get image paths before deleting
        cursor.execute("SELECT image_path FROM driver_encodings WHERE driver_ref_id = ?", (driver_db_id,))
        image_paths = cursor.fetchall()
        # Delete driver (CASCADE deletes encodings)
        cursor.execute("DELETE FROM drivers WHERE id = ?", (driver_db_id,))
        conn.commit()

        if cursor.rowcount > 0:
            print(f"Driver {driver_id} deleted successfully from DB.")
            # Delete image files
            deleted_files_count = 0
            for img_tuple in image_paths:
                img_path = img_tuple[0]
                if img_path and os.path.exists(img_path):
                    try: os.remove(img_path); deleted_files_count += 1
                    except OSError as e: print(f"Error deleting image file {img_path}: {e}")
            print(f"Deleted {deleted_files_count} associated image files.")
            return True
        else: return False # Deletion failed in DB

    except sqlite3.Error as e:
        print(f"Database Error deleting driver: {e}")
        if conn: conn.rollback()
        return False
    except Exception as e:
         print(f"Unexpected error in delete_driver_db: {e}")
         if conn: conn.rollback()
         return False
    finally:
        if conn:
            conn.close()


def load_all_driver_encodings():
    """Loads all encodings from the driver_encodings table, joining with driver info."""
    known_encodings = []
    conn = None
    drivers_found = set()
    try:
        conn = sqlite3.connect(config.DB_FILE)
        cursor = conn.cursor()
        sql = """
            SELECT d.driver_id, d.name, de.encoding
            FROM drivers d JOIN driver_encodings de ON d.id = de.driver_ref_id
        """
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            driver_id, name, encoding_blob = row
            encoding = np.frombuffer(encoding_blob, dtype=np.float32)
            known_encodings.append((driver_id, name, encoding))
            drivers_found.add(driver_id)
        print(f"Loaded {len(known_encodings)} encodings for {len(drivers_found)} unique drivers.")
        return known_encodings # Return the loaded list

    except sqlite3.Error as e:
        print(f"Database Error loading encodings: {e}")
        return [] # Return empty list on error
    except Exception as e:
         print(f"Unexpected error in load_all_driver_encodings: {e}")
         return []
    finally:
        if conn:
            conn.close()

# --- search_driver_db ---
# (This function might need revision depending on exact needs,
# the version in run_driver_manager_ui queries directly for now)
# If needed, add it here similar to the others.