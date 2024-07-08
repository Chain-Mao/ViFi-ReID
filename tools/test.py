import os
import shutil

def move_and_delete_images(directory):
    """
    Move all images from subfolders in the specified directory to the directory itself.
    Then delete the subfolders.

    :param directory: The directory to process.
    """

    # Check if the specified directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # List all subdirectories
    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # Process each subdirectory
    for subdirectory in subdirectories:
        # List all files in the subdirectory
        for file in os.listdir(subdirectory):
            # Check if the file is an image
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                file_path = os.path.join(subdirectory, file)
                # Move the file to the main directory
                shutil.move(file_path, directory)

        # Delete the subdirectory
        os.rmdir(subdirectory)

    print(f"All images moved from subdirectories to '{directory}', and subdirectories deleted.")

# Example usage
directory_path = "/data1/fast-reid/datasets/Market-ViFi/train/vision"  # Replace with your directory path
move_and_delete_images(directory_path)
