import os

data_path = "genres_original"

if not os.path.exists(data_path):
    print("Bhai Dataset nahi mil raha")

else:
    print("We found the datset.......")

    for genre_folder in sorted(os.listdir(data_path)):
        genre_path = os.path.join(data_path, genre_folder)
        if os.path.isdir(genre_path):
            
            # If it is a directory, we list its contents to get all the .wav file names.
            files_in_genre = os.listdir(genre_path)
            
            # The number of files is simply the length of this list.
            number_of_files = len(files_in_genre)
            
            # We print the result in a nicely formatted way.
            # The .ljust(12) method pads the string with spaces to a length of 12,
            # which helps to align the output neatly in columns.
            print(f"Genre: {genre_folder.ljust(12)} | File Count: {number_of_files}")