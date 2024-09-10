import os
import re
import struct
import json
import safetensors

def read_safetensors_json(file_path):
    """Reads the metadata (JSON header) from a safetensors file."""
    with open(file_path, 'rb') as f:
        # Step 1: Read the first 8 bytes to get the size of the header (N)
        header_size_bytes = f.read(8)
        header_size = struct.unpack('Q', header_size_bytes)[0]  # 'Q' is for unsigned 64-bit integer

        # Step 2: Read the next N bytes which contain the JSON header
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode('utf-8')

        # Step 3: Parse the JSON header
        header = json.loads(header_str)
        return header

class SafetensorsWrapper:
    def __init__(self, file_paths, framework="pt", **kv):
        """
        Initialize the SafetensorsWrapper with one or multiple safetensors files.

        Args:
            file_paths (str or list of str): Paths to the safetensors file(s).
            framework (str): The framework to load the tensors for (e.g., "pt" for PyTorch).
        """
        # Ensure file_paths is a list, even if a single file is provided
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        self._file_paths = []
        # Dictionary to hold tensor references from multiple files
        self._tensors = {}
        
        # Load tensors from each file and merge them into self._tensors
        for file_path in file_paths:
            self._load_file(file_path, framework, **kv)

    def _load_file(self, file_path, framework, **kv):
        """Helper function to load a safetensors file and merge its tensors."""
        file = safetensors.safe_open(file_path, framework=framework, **kv)
        for key in file.keys():
            if key in self._tensors:
                raise ValueError(f"Tensor name collision detected: '{key}' is already loaded from another file.")
            self._tensors[key] = file
        self._file_paths.append(file_path)

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def keys(self):
        """Returns the list of tensor names across all wrapped files."""
        return list(self._tensors.keys())

    def items(self):
        """Lazy generator that yields (name, tensor) pairs."""
        for key in self.keys():
            yield key, self._tensors[key].get_tensor(key)

    def slices(self):
        """Lazy generator that yields (name, slice) pairs."""
        for key in self.keys():
            yield key, self._tensors[key].get_slice(key)

    def __contains__(self, key):
        """Check if a tensor exists across any of the wrapped files."""
        return key in self._tensors

    def __getitem__(self, key):
        """Get a tensor by its name from any of the wrapped files."""
        if key in self._tensors:
            return self._tensors[key].get_tensor(key)
        raise KeyError(f"Tensor '{key}' not found in any wrapped files.")

    def __len__(self):
        """Return the total number of unique tensors across all wrapped files."""
        return len(self._tensors)

    def __iter__(self):
        """Return an iterator over the tensor names."""
        return iter(self.keys())

    def __repr__(self):
        """Return a string representation of the object."""
        return f"<SafetensorsWrapper: {len(self)} tensors from {len(self._tensors)} files>"

    def get_index(self):
        """
        Generate an index that includes the total size and the weight map of tensors.

        Returns:
            dict: A dictionary containing the metadata, total size, and weight map.
        """
        weight_map = {}
        total_size = 0
        
        # Loop through each file and gather the metadata
        for file_path in self._file_paths:
            metadata = read_safetensors_json(file_path)
            file_name = os.path.basename(file_path)
            
            # Aggregate metadata to form the weight map and calculate the total size
            for tensor_name, tensor_info in metadata.items():
                if tensor_name == '__metadata__':
                    continue  # Skip the metadata entry
                weight_map[tensor_name] = file_name
                total_size += tensor_info['data_offsets'][1] - tensor_info['data_offsets'][0]

        # Create the final index structure
        index = {
            "metadata": {
                "total_size": total_size
            },
            "weight_map": weight_map
        }
        return index


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CLI tool for working with Safetensors files"
    )
    
    subparsers = parser.add_subparsers(dest="command")

    # Command to read metadata
    parser_metadata = subparsers.add_parser('metadata', help="Read Safetensors metadata")
    parser_metadata.add_argument('file_path', type=str, help="Path to the Safetensors file")

    # Command to list tensors
    parser_list = subparsers.add_parser('list', help="List tensors from Safetensors files")
    parser_list.add_argument('file_paths', nargs='+', type=str, help="One or more Safetensors file paths")

    # Command to generate index
    parser_index = subparsers.add_parser('index', help="Generate index for Safetensors files")
    parser_index.add_argument('file_paths', nargs='+', type=str, help="One or more Safetensors file paths")

    args = parser.parse_args()

    def read_metadata(file_path):
        """Read and display the JSON metadata of a Safetensors file."""
        try:
            metadata = read_safetensors_json(file_path)
            print(json.dumps(metadata, indent=2))
        except BrokenPipeError:
            sys.stdout.close()
            sys.exit(1)
    
    def list_tensors(file_paths):
        """List all tensors from one or more Safetensors files."""
        try:
            wrapper = SafetensorsWrapper(file_paths)
            for tensor_name in wrapper.keys():
                print(tensor_name)
        except BrokenPipeError:
            sys.stdout.close()
            sys.exit(1)
    
    def generate_index(file_paths):
        """Generate and display the index (metadata and weight map) for Safetensors files."""
        try:
            wrapper = SafetensorsWrapper(file_paths)
            index = wrapper.get_index()
            print(json.dumps(index, indent=2))
        except BrokenPipeError:
            sys.stdout.close()
            sys.exit(1)
    
    try:
        if args.command == 'metadata':
            read_metadata(args.file_path)
        elif args.command == 'list':
            list_tensors(args.file_paths)
        elif args.command == 'index':
            generate_index(args.file_paths)
        else:
            parser.print_help()
    except BrokenPipeError:
        sys.stdout.close()
        sys.exit(1)

if __name__ == "__main__":
    main()
