%Label format for darknet : <object-class-id> <x-centre> <y-centre> <width> <height>
% videos: 48, labelled frames per video: 100
% images resolution: 1280x720 (width = 1280, height = 720)

data = load('metadata.mat');
videos = data.video;
n_frames = 100;
frames = [];
n_videos = 48;
frame_width = 1280;
frame_height = 720;

count = 1;

for i=1:n_videos

    video = videos(i);

    for j=1:n_frames
        
        bounding_box = getBoundingBoxes(video,j); %x y bounding_box_width bounding_box_height
        rows_to_remove = []; %to clean rows of 0s in bounding_box

        for k=1:size(bounding_box,1)
            if bounding_box(k,:) == zeros(1,4)
                rows_to_remove = [rows_to_remove, k];
            end
        end

        bounding_box(rows_to_remove,:) = [];
        
        object_class_id = zeros(size(bounding_box,1),1);

        x_centre = (2 * bounding_box(:,1) + bounding_box(:,3))/ (2 * frame_width); %x centers of bounding boxes
        y_centre = (2 * bounding_box(:,2) + bounding_box(:,4))/ (2 * frame_height);%y centers of bounding boxes
        width = bounding_box(:,3) / frame_width;
        height = bounding_box(:,4) / frame_height;

        label = [object_class_id, x_centre, y_centre, width, height];

        img_label = strcat('image_',string(count));
        count = count + 1;
        path = strcat('images/', img_label);
        writematrix(label, path, 'Delimiter', 'space');

    end

end