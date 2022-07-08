 n_samples = 4800;
 n_train = round(0.7 * n_samples);

 random_permutation = randperm(n_samples);

 count = 1;
    
 for i=1:size(random_permutation,2)

     sample_index = random_permutation(i);
     line = strcat('custom_data/images/image_', string(sample_index), '.jpg');

     if count <= n_train
         writelines(line, 'train.txt', 'WriteMode','append', 'Encoding','UTF-8');
     else
         writelines(line, 'test.txt', 'WriteMode','append', 'Encoding','UTF-8');
     end

     count = count + 1;

 end