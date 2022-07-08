partial_path = "_LABELLED_SAMPLES";
main_dir = dir(partial_path);


count = 1;

for i=3:size(main_dir,1)

    current_directory = main_dir(i);
    current_path = strcat(partial_path, "/", current_directory.name);

    current_folder = dir(current_path);


    for j=3:size(current_folder,1) - 1

        frame_name = current_folder(j).name;
        frame = imread(strcat(current_path,"/",frame_name));
         
        new_frame_name = "image_" + string(count) + ".jpg";
        path = strcat("images/", new_frame_name);
        imwrite(frame, path);

         count = count + 1;

    end
  
end