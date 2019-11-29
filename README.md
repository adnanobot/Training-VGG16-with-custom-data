# Training-VGG16-with-custom-data

## Downloading annotated data from Google's "Open Images V4"
  
  1. Clone the repository https://github.com/EscVM/OIDv4_ToolKit: 
      'git clone https://github.com/EscVM/OIDv4_ToolKit.git'
  
  2. Download data: 
      'python main.py downloader --classes Screwdriver Eraser --type_csv train'
      
      *for validation and test data replace train by test/validation
      
  3. Easy way to see 
  [here](https://www.learnopencv.com/fast-image-downloader-for-open-images-v4/)
  which of types of objects are available and how many instances. 
    
Reference: 
 [1] Medium
 [blog](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwiXz4-3q4_mAhUwyKYKHX-GBMcQFjAAegQIAhAB&url=https%3A%2F%2Fmedium.com%2F%40c.n.veeraganesh%2Fhow-to-prepare-your-own-customized-dataset-using-open-images-dataset-v4-8dfce9b9e147&usg=AOvVaw3Wt7AUJEl8rxvjDGQCAMNZ) on how to get the data.
 
 [2] Youtube 
 [tutorial](https://www.youtube.com/watch?v=oDHpqu52soI&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL&index=13)
 on how to train/fine tune VGG16 for custom data.
 
 
