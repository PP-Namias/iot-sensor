# ui_manager.py
import customtkinter as ctk
import os
import time
import shutil
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox
import config
from database_manager import add_driver_db, delete_driver_db
from face_recognizer import get_face_encoding_from_image

def run_driver_manager_ui(face_detector_net, face_embedder_net):
    """Runs the customtkinter interface for managing drivers."""
    
    # Check if models were loaded successfully before starting UI
    if not face_detector_net or not face_embedder_net:
        messagebox.showerror("Model Load Error", 
                            "Error: Face recognition models failed to load.\n"
                            "Cannot add drivers.\n"
                            "Please check model files and restart.")
        return  # Exit UI if models aren't ready
    
    # Set up the app theme
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
    
    root = ctk.CTk()
    root.title("Truck Driver Management")
    root.geometry("700x550")
    
    # Maximize the window on startup
    root.state("zoomed")
    
    # Create a frame for inputs
    input_frame = ctk.CTkFrame(root)
    input_frame.pack(pady=10, padx=10, fill="x")
    
    # Create title
    title_label = ctk.CTkLabel(input_frame, text="Driver Manager", font=ctk.CTkFont(size=20, weight="bold"))
    title_label.grid(row=0, column=0, columnspan=3, pady=10, padx=10, sticky="w")
    
    # Driver ID input
    driver_id_label = ctk.CTkLabel(input_frame, text="Driver ID/Name:")
    driver_id_label.grid(row=1, column=0, pady=10, padx=10, sticky="w")
    
    driver_id_entry = ctk.CTkEntry(input_frame, width=200)
    driver_id_entry.grid(row=1, column=1, pady=10, padx=10, sticky="w")
    
    # Image paths input
    image_paths_label = ctk.CTkLabel(input_frame, text="Image Files:")
    image_paths_label.grid(row=2, column=0, pady=10, padx=10, sticky="w")
    
    image_paths_entry = ctk.CTkEntry(input_frame, width=300, state="readonly")
    image_paths_entry.grid(row=2, column=1, pady=10, padx=10, sticky="w")
    
    # Function to handle file browsing
    def browse_files():
        filetypes = (("Image Files", "*.jpg *.jpeg *.png"),)
        files = filedialog.askopenfilenames(initialdir=os.getcwd(), filetypes=filetypes)
        if files:
            image_paths_entry.configure(state="normal")
            image_paths_entry.delete(0, tk.END)
            image_paths_entry.insert(0, ";".join(files))
            image_paths_entry.configure(state="readonly")
    
    browse_button = ctk.CTkButton(input_frame, text="Browse (Select 1-3 Images)", command=browse_files)
    browse_button.grid(row=2, column=2, pady=10, padx=10, sticky="w")
    
    # Create a frame for buttons
    button_frame = ctk.CTkFrame(root)
    button_frame.pack(pady=10, padx=10, fill="x")
    
    # Status display
    status_label = ctk.CTkLabel(button_frame, text="Status:")
    status_label.grid(row=0, column=0, pady=10, padx=10, sticky="w")
    
    status_display = ctk.CTkLabel(button_frame, text="Ready", width=500)
    status_display.grid(row=0, column=1, columnspan=3, pady=10, padx=10, sticky="w")
    
    # Output text area
    output_frame = ctk.CTkFrame(root)
    output_frame.pack(pady=10, padx=10, fill="both", expand=True)
    
    output_text = ctk.CTkTextbox(output_frame, width=680, height=250)
    output_text.pack(pady=10, padx=10, fill="both", expand=True)
    
    # Custom print function to output to textbox
    def print_to_output(message):
        output_text.configure(state="normal")
        output_text.insert(tk.END, message + "\n")
        output_text.see(tk.END)
        output_text.configure(state="disabled")
        root.update_idletasks()
    
    # Handle adding a driver
    def add_driver():
        driver_id = driver_id_entry.get().strip()
        image_paths_str = image_paths_entry.get()
        
        print_to_output(f"\nAttempting to add driver: {driver_id}")
        
        if not driver_id:
            status_display.configure(text="Error: Driver ID/Name required.")
            print_to_output("Add failed: Missing ID.")
            return
            
        if not image_paths_str:
            status_display.configure(text="Error: Select image(s).")
            print_to_output("Add failed: Missing images.")
            return
        
        image_paths_list = [p for p in image_paths_str.split(';') if p]
        if not image_paths_list:
            status_display.configure(text="Error: No valid paths selected.")
            print_to_output("Add failed: Invalid paths.")
            return
        
        valid_paths, all_valid = [], True
        for img_path in image_paths_list:
            if not os.path.exists(img_path):
                status_display.configure(text=f'Error: Not found "{img_path}"')
                all_valid = False
                break
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                status_display.configure(text=f'Error: Bad format "{os.path.basename(img_path)}"')
                all_valid = False
                break
            valid_paths.append(img_path)
        
        if not all_valid:
            print_to_output("Add failed: File validation error.")
            return
        
        # --- Process Images ---
        image_encoding_pairs = []
        processed_files = []
        status_display.configure(text="Processing images...")
        root.update_idletasks()
        success = True
        
        for i, original_path in enumerate(valid_paths):
            safe_id = "".join(c for c in driver_id if c.isalnum()).rstrip()
            _, ext = os.path.splitext(original_path)
            fname = f"{safe_id}_{i+1}_{int(time.time())}{ext}"
            dest_path = os.path.join(config.IMAGE_DIR, fname)
            print_to_output(f" Processing: {os.path.basename(original_path)} -> {fname}")
            
            try:
                shutil.copy(original_path, dest_path)
                processed_files.append(dest_path)
                encoding = get_face_encoding_from_image(dest_path, face_detector_net, face_embedder_net)
                if encoding is not None:
                    image_encoding_pairs.append((dest_path, encoding))
                    print_to_output(f" -> Encoding OK")
                else:
                    print_to_output(f" -> Encoding FAILED. Skipping.")
            except Exception as e:
                status_display.configure(text=f'Error processing {os.path.basename(original_path)}: {e}')
                print_to_output(f"Error during processing: {e}")
                success = False
                if os.path.exists(dest_path) and dest_path in processed_files:
                    try:
                        os.remove(dest_path)
                        processed_files.remove(dest_path)
                    except OSError:
                        pass
                break
        
        # --- Add to DB ---
        if success and image_encoding_pairs:
            status_display.configure(text="Adding to database...")
            root.update_idletasks()
            if add_driver_db(driver_id, driver_id, image_encoding_pairs):
                print_to_output(f"Driver '{driver_id}' added/updated with {len(image_encoding_pairs)} encodings.")
                status_display.configure(text="Driver Added/Updated Successfully!")
                driver_id_entry.delete(0, tk.END)
                image_paths_entry.configure(state="normal")
                image_paths_entry.delete(0, tk.END)
                image_paths_entry.configure(state="readonly")
            else:
                status_display.configure(text="Error adding driver to DB (see output).")
                print_to_output("DB Add failed. Cleaning up processed images...")
                for f in processed_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        elif success and not image_encoding_pairs:
            status_display.configure(text="Error: Encoding failed for all images.")
            print_to_output("Add failed: No encodings generated. Cleaning up images...")
            for f in processed_files:
                try:
                    os.remove(f)
                except OSError:
                    pass
        elif not success:
            print_to_output("Add failed due to processing error. Check status/output.")
    
    # Handle searching for a driver
    def search_driver():
        driver_id = driver_id_entry.get().strip()
        
        print_to_output(f"\nSearching for driver: {driver_id}")
        
        if not driver_id:
            status_display.configure(text="Error: Enter Driver ID/Name.")
            print_to_output("Search failed: Missing ID.")
            return
        
        conn = None
        try:
            conn = sqlite3.connect(config.DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT id, driver_id, name FROM drivers WHERE driver_id = ?", (driver_id,))
            result = cursor.fetchone()
            
            if result:
                db_id, found_id, found_name = result
                print_to_output(f"Found: ID={found_id}, Name={found_name} (DB ID: {db_id})")
                cursor.execute("SELECT image_path FROM driver_encodings WHERE driver_ref_id = ?", (db_id,))
                images = cursor.fetchall()
                print_to_output(f"  Associated Images ({len(images)}):")
                for img_tuple in images:
                    print_to_output(f"  - {img_tuple[0]}")
                status_display.configure(text="Driver found.")
            else:
                print_to_output(f"Driver ID '{driver_id}' not found.")
                status_display.configure(text="Driver not found.")
        except sqlite3.Error as e:
            status_display.configure(text=f"DB Search Error: {e}")
            print_to_output(f"Search error: {e}")
        finally:
            if conn:
                conn.close()
    
    # Handle deleting a driver
    def delete_driver():
        driver_id = driver_id_entry.get().strip()
        
        print_to_output(f"\nAttempting to delete driver: {driver_id}")
        
        if not driver_id:
            status_display.configure(text="Error: Enter Driver ID/Name.")
            print_to_output("Delete failed: Missing ID.")
            return
        
        confirm = messagebox.askyesno("Confirm Deletion", 
                                      f'DELETE driver "{driver_id}"?\n'
                                      f'This removes the driver and ALL associated images/encodings.')
        
        if confirm:
            print_to_output(f"Deleting driver {driver_id}...")
            if delete_driver_db(driver_id):
                print_to_output(f"Deletion successful.")
                status_display.configure(text="Driver Deleted.")
                driver_id_entry.delete(0, tk.END)
            else:
                print_to_output(f"Deletion failed (check output/logs).")
                status_display.configure(text="Error deleting driver (see output).")
        else:
            print_to_output("Deletion cancelled.")
    
    # Create action buttons
    action_frame = ctk.CTkFrame(root)
    action_frame.pack(pady=10, padx=10, fill="x")
    
    add_button = ctk.CTkButton(action_frame, text="Add Driver", command=add_driver)
    add_button.grid(row=0, column=0, pady=10, padx=10)
    
    search_button = ctk.CTkButton(action_frame, text="Search Driver", command=search_driver)
    search_button.grid(row=0, column=1, pady=10, padx=10)
    
    delete_button = ctk.CTkButton(action_frame, text="Delete Driver", command=delete_driver)
    delete_button.grid(row=0, column=2, pady=10, padx=10)
    
    exit_button = ctk.CTkButton(action_frame, text="Exit Manager", command=root.destroy)
    exit_button.grid(row=0, column=3, pady=10, padx=10)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start the main loop
    root.mainloop()