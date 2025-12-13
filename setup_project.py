import os

project_structure = {
    "__pycache__/",
    ".venv/",
    "chrome_langchain_db/",
    "model/",
    "templates/",
    "app.py",
    "main.py",
    "requirements.txt",
    "vector.py"
}

# Function to create folders and files
def create_project_structure(structure):
    for folder, contents in structure.items():
        os.makedirs(folder, exist_ok=True)
        for item in contents:
            path = os.path.join(folder, item)
            if item.endswith("/"):  # Create subfolder
                os.makedirs(path, exist_ok=True)
                print(f"Created folder: {path}")
            else:  # Create empty file
                open(path, 'a').close()
                print(f"Created file: {path}")

# Run the setup
if __name__ == "__main__":
    create_project_structure(project_structure)
    print("\nProject structure created successfully!")
