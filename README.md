## Exercise 1 

+ Run exercise-1.py file in src/exercises/;
+ It applies system represented by the image located in mod dir, with an example document in file system.

## Exercise 2

+ Run exercise-2.py file in src/exercises/;
+ It applies the same system but with variants trying to improve results; 
+ The results from exercise-1 and this exercise are represented in a txt file localed in results dir and it uses a dataset with 1500 abstract documents.

## Exercise 4

+ Tornado package is necessary;
+ Conection to internet is necessary to get RSS file;
+ Run exercise-4.py in src/exercises/ to start web server;
+ Open web browser and do a request to "localhost:8888" or if you changed the port "localhost:newPort"
+ This request return an html page which allows to see every sport article and its key phrases. Also, it has filter mechanisms. 

## Extra files

+ During the development, we generated a pickle file for word vectors that exceed the max unique file size. The big_files_url contains an URL to download that file. It should be saved into src/files/ and the name must be preserved or you might change on code. 
