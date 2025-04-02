# arduino_comm.py
import serial
import time
import config # Use constants from config

class ArduinoComm:
    def __init__(self, port=config.ARDUINO_PORT, baudrate=config.ARDUINO_BAUDRATE, timeout=config.ARDUINO_TIMEOUT):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.arduino = None
        self.connected = False
        # Disable if port is not set
        if not self.port:
             print("Arduino port not configured in config.py. Disabling Arduino communication.")

    def connect(self):
        """Attempts to connect to the Arduino."""
        if not self.port: # Don't attempt if port is None/empty
            self.connected = False
            return False
        try:
            time.sleep(0.5)
            self.arduino = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout)
            self.connected = True # Assume connected initially
            print(f"Attempting connection to Arduino on {self.port}...")
            time.sleep(2.5) # Allow time for Arduino reset
            success = self.send('0') # Send initial 'off' command
            time.sleep(0.1)
            if success:
                 print("Successfully connected and communicated with Arduino.")
                 return True
            else:
                 # If initial send fails, connection likely didn't establish properly
                 print("Connected to port, but failed initial communication.")
                 self.disconnect() # Clean up
                 return False
        except serial.SerialException as e:
            self.connected = False
            print(f"Failed to connect to Arduino on {self.port}: {e}")
            self.arduino = None
            return False
        except Exception as e:
             self.connected = False
             print(f"An unexpected error occurred during Arduino initialization: {e}")
             self.arduino = None
             return False

    def send(self, command):
        """Sends a command ('0' or '1') to the Arduino."""
        if not self.connected or not self.arduino:
            # print("Debug: Arduino not connected, cannot send command.") # Optional debug
            return False
        try:
            bytes_written = self.arduino.write(command.encode())
            # print(f"Sent '{command}' to Arduino ({bytes_written} bytes).") # Optional debug
            return True # Assume success if write doesn't raise error
        except serial.SerialException as e:
            print(f"Arduino write error: {e}. Disconnecting.")
            self.disconnect() # Disconnect on write error
            return False
        except Exception as e:
             print(f"Unexpected error sending command to Arduino: {e}")
             self.disconnect()
             return False

    def disconnect(self):
        """Closes the serial connection."""
        if self.arduino and self.arduino.is_open:
            try:
                # Attempt to send '0' before closing, but don't rely on it
                # self.send('0') # Might fail if connection is already broken
                # time.sleep(0.1)
                self.arduino.close()
                print("Arduino connection closed.")
            except Exception as e:
                print(f"Error closing Arduino connection: {e}")
        self.connected = False
        self.arduino = None # Ensure arduino object is None after disconnect

    def is_connected(self):
        """Returns the connection status."""
        return self.connected