def capitalize_words_in_file(file_path):
    try:
        with open(file_path, "r") as file:
            lines = [line.strip().lower() for line in file]

        with open(file_path, "w") as file:
            file.write("\n".join(lines))

        print(f"Successfully capitalized and updated {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


file_path = "./data/medical_alpha.txt"
capitalize_words_in_file(file_path)
