# ui_manager.py
import PySimpleGUI as sg
import os
import time
import shutil
import sqlite3 # Need for direct DB query in Search
import config # Import configuration
# Import necessary functions from other modules
from database_manager import add_driver_db, delete_driver_db
from face_recognizer import get_face_encoding_from_image

# Note: This function now needs the loaded models passed to it
def run_driver_manager_ui(face_detector_net, face_embedder_net):
    """Runs the PySimpleGUI interface for managing drivers."""
    sg.theme('SystemDefault')

    # Check if models were loaded successfully before starting UI
    if not face_detector_net or not face_embedder_net:
         sg.popup_error("Error: Face recognition models failed to load.\nCannot add drivers.\nPlease check model files and restart.", title="Model Load Error")
         return # Exit UI if models aren't ready

    layout = [
        [sg.Text('Driver Manager', font=('Helvetica', 16))],
        [sg.Text('Driver ID/Name:', size=(15,1)), sg.InputText(key='-DRIVER_ID-')],
        [sg.Text('Image Files:', size=(15,1)),
         sg.Input(key='-IMAGE_PATHS-', enable_events=True, readonly=True),
         sg.FilesBrowse("Browse (Select 1-3 Images: Front, Left, Right)",
                        file_types=(("Image Files", "*.jpg;*.png"),),
                        initial_folder=os.getcwd(),
                        files_delimiter=';')
        ],
        [sg.Button('Add Driver'), sg.Button('Search Driver'), sg.Button('Delete Driver')],
        [sg.Text('Status:', size=(15,1)), sg.Text('', size=(50,1), key='-STATUS-')],
        [sg.Multiline(size=(70, 10), key='-OUTPUT-', disabled=True, autoscroll=True, reroute_stdout=True, reroute_stderr=True)], # Reroute print/errors
        [sg.Button('Exit Manager')]
    ]

    window = sg.Window('Truck Driver Management', layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == 'Exit Manager':
            break

        driver_id = values['-DRIVER_ID-'].strip()
        image_paths_str = values['-IMAGE_PATHS-']
        window['-STATUS-'].update('') # Clear status on new action

        if event == 'Add Driver':
            print(f"\nAttempting to add driver: {driver_id}") # Use print now routed to Multiline
            if not driver_id: window['-STATUS-'].update('Error: Driver ID/Name required.'); print("Add failed: Missing ID."); continue
            if not image_paths_str: window['-STATUS-'].update('Error: Select image(s).'); print("Add failed: Missing images."); continue

            image_paths_list = [p for p in image_paths_str.split(';') if p]
            if not image_paths_list: window['-STATUS-'].update('Error: No valid paths selected.'); print("Add failed: Invalid paths."); continue

            valid_paths, all_valid = [], True
            for img_path in image_paths_list:
                 if not os.path.exists(img_path): window['-STATUS-'].update(f'Error: Not found "{img_path}"'); all_valid = False; break
                 if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')): window['-STATUS-'].update(f'Error: Bad format "{os.path.basename(img_path)}"'); all_valid = False; break
                 valid_paths.append(img_path)
            if not all_valid: print("Add failed: File validation error."); continue

            # --- Process Images ---
            image_encoding_pairs = []
            processed_files = []
            window['-STATUS-'].update('Processing images...'); window.refresh()
            success = True
            for i, original_path in enumerate(valid_paths):
                safe_id = "".join(c for c in driver_id if c.isalnum()).rstrip()
                _, ext = os.path.splitext(original_path)
                fname = f"{safe_id}_{i+1}_{int(time.time())}{ext}"
                dest_path = os.path.join(config.IMAGE_DIR, fname)
                print(f" Processing: {os.path.basename(original_path)} -> {fname}")

                try:
                    shutil.copy(original_path, dest_path)
                    processed_files.append(dest_path) # Track successful copies
                    encoding = get_face_encoding_from_image(dest_path, face_detector_net, face_embedder_net) # Pass models
                    if encoding is not None:
                        image_encoding_pairs.append((dest_path, encoding))
                        print(f" -> Encoding OK")
                    else:
                        print(f" -> Encoding FAILED. Skipping.")
                        # Keep file for now, maybe user wants to retry? Or delete:
                        # try: os.remove(dest_path); processed_files.remove(dest_path) except OSError: pass
                except Exception as e:
                    window['-STATUS-'].update(f'Error processing {os.path.basename(original_path)}: {e}')
                    print(f"Error during processing: {e}")
                    success = False
                    # Clean up file copied in this iteration if error occurred
                    if os.path.exists(dest_path) and dest_path in processed_files:
                         try: os.remove(dest_path); processed_files.remove(dest_path) 
                         
                         except OSError: pass
                    break # Stop processing more files for this driver on error

            # --- Add to DB ---
            if success and image_encoding_pairs:
                window['-STATUS-'].update('Adding to database...'); window.refresh()
                # Use driver_id as name for simplicity
                if add_driver_db(driver_id, driver_id, image_encoding_pairs):
                    print(f"Driver '{driver_id}' added/updated with {len(image_encoding_pairs)} encodings.")
                    window['-STATUS-'].update('Driver Added/Updated Successfully!')
                    window['-DRIVER_ID-'].update(''); window['-IMAGE_PATHS-'].update('') # Clear inputs
                else:
                    window['-STATUS-'].update('Error adding driver to DB (see output).')
                    print("DB Add failed. Cleaning up processed images...")
                    for f in processed_files: 
                        try: os.remove(f) 
                        except OSError: pass
            elif success and not image_encoding_pairs:
                 window['-STATUS-'].update('Error: Encoding failed for all images.')
                 print("Add failed: No encodings generated. Cleaning up images...")
                 for f in processed_files: 
                    try: os.remove(f) 
                    except OSError: pass
            elif not success:
                print("Add failed due to processing error. Check status/output.")
                # Some files might have been cleaned up already in the loop

        elif event == 'Search Driver':
            print(f"\nSearching for driver: {driver_id}")
            if not driver_id: window['-STATUS-'].update('Error: Enter Driver ID/Name.'); print("Search failed: Missing ID."); continue
            conn = None
            try:
                conn = sqlite3.connect(config.DB_FILE)
                cursor = conn.cursor()
                cursor.execute("SELECT id, driver_id, name FROM drivers WHERE driver_id = ?", (driver_id,))
                result = cursor.fetchone()
                if result:
                    db_id, found_id, found_name = result
                    print(f"Found: ID={found_id}, Name={found_name} (DB ID: {db_id})")
                    cursor.execute("SELECT image_path FROM driver_encodings WHERE driver_ref_id = ?", (db_id,))
                    images = cursor.fetchall()
                    print(f"  Associated Images ({len(images)}):")
                    for img_tuple in images: print(f"  - {img_tuple[0]}")
                    window['-STATUS-'].update('Driver found.')
                else:
                    print(f"Driver ID '{driver_id}' not found.")
                    window['-STATUS-'].update('Driver not found.')
            except sqlite3.Error as e: window['-STATUS-'].update(f'DB Search Error: {e}'); print(f"Search error: {e}")
            finally:
                if conn: conn.close()

        elif event == 'Delete Driver':
            print(f"\nAttempting to delete driver: {driver_id}")
            if not driver_id: window['-STATUS-'].update('Error: Enter Driver ID/Name.'); print("Delete failed: Missing ID."); continue
            confirm = sg.popup_yes_no(f'DELETE driver "{driver_id}"?\nThis removes the driver and ALL associated images/encodings.', title='Confirm Deletion')
            if confirm == 'Yes':
                print(f"Deleting driver {driver_id}...")
                if delete_driver_db(driver_id): # Use the updated function
                    print(f"Deletion successful.")
                    window['-STATUS-'].update('Driver Deleted.')
                    window['-DRIVER_ID-'].update('') # Clear ID field
                else:
                    print(f"Deletion failed (check output/logs).")
                    window['-STATUS-'].update('Error deleting driver (see output).')
            else:
                print("Deletion cancelled.")

    window.close()