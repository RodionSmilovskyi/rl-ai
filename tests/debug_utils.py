import os
import sys
import datetime
import atexit

class InferenceDebugger:
    def __init__(self, model_path, args):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.debug_dir = os.path.join(self.project_root, 'debugging')
        
        # Create debugging directory if it doesn't exist
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Format filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_basename = os.path.basename(model_path).replace('.', '_')
        filename = f"{model_basename}_{timestamp}.log"
        self.filepath = os.path.join(self.debug_dir, filename)
        
        self.log_file = open(self.filepath, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        sys.stdout = self
        sys.stderr = self
        
        # Write header
        self.write_header(model_path, args)

        # Register cleanup to restore original stdout/stderr and close file
        atexit.register(self.close)

    def write_header(self, model_path, args):
        header = f"=== Inference Run ===\n"
        header += f"Model: {model_path}\n"
        header += f"Arguments: {args}\n"
        header += f"Timestamp: {datetime.datetime.now().isoformat()}\n"
        header += f"=====================\n\n"
        self.log_file.write(header)
        self.original_stdout.write(header)
        self.log_file.flush()

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_file.close()
