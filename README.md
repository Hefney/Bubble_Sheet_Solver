# Bubble Sheet Solver
A college project on extracting the code and answers marked by a student on a bubble sheet.
# Algorithm steps
## Extract The paper from the image
We find the contours in an image -after making some preprocessing as Closing, Thresholding-, and detect the largest one, this is paper, then we get the 4 points of The paper from this contour by approximating, and apply perspective transform on the image.
## Standard Hough Circles 
We use Hough Circles to detect all the circles in the image, we identify min and max radius to search within, in order not to some of the letters as circles, such as: 'o', 'D' 
## Comparing the Circles with a certain threshold
We iterate over every Circle -after thresholding the whole paper- by a filter of ones then we get the AND of the filter with the circle.
If the Circle is so bright -> hence it's not Marked.
else -> it's marked
# Results
You can see in the Results, That we detected the Code circles as RED circles and the Questions circles as BLUE ones, The marked ones are Dilated
