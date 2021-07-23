# MouseCTRL: Mouse control. Simplified.
Our app eases the navigation of the mouse involved in tasks such as writing something using a touchpad or playing games. We have come up with an efficient and easier alternative for the traditional touchpad or a mouse. The navigation is facilitated by hand gestures which are in turn recognized by a camera. It then gives an input to the machine to move the mouse to the place corresponding to the position of the hand of the user. This process is automated and quite fast. We have used Artificial Intelligence to recognize the hand of the user which then earmarks specific points on their hand. This is followed by the tracking of the position of those points by the AI mechanism. This finally leads to the movement of the mouse.

![Hand Recognition](https://raw.githubusercontent.com/MouseCTRL/MouseCTRL/main/images/Hand-Recognition.png)

For smooth movement of the mouse we consider the movement of the point 8 and 12 together. The movement of point 8 is considered for a left-click event while the movement of point 12 is considered for a right-click event. Point 16 is used in an event featuring continuous right-click with a left click. Finally, Point 20 is used for a middle click event.
Main uses of our app: 
-	Online teaching purposes: Our app can easily facilitate writing on virtual whiteboard apps such as microsoft paint or onenote.
-	Gaming Purposes: Our app can be used for gaming purposes, especially in games which require a change in direction or virtual movement.
-	Day to day usage: Finally, our app can be used for day-to-day purposes which require a device. It can replace the normal mouse require for such tasks.

To download our app:
- Download the setup file from the following link:
```bash
https://www.dropbox.com/s/rvw1yrzdy4tsgg0/MouseCTRL-Setup.exe?dl=0
```
- Then, run the File: `MouseCTRL-Setup.exe` from the Downloads folder
- Choose the app installation location and click Next
- The Installer will download the App, and the required dependencies to run the App
- After the installation finishes, click close and Search for `MouseCTRL` in the Search Bar and Run the `MouseCTRL.exe`

## Note
Before running the code, 
- Install the dependencies: 
```python
pip install opencv-python mediapipe autopy
python >= 3.8
```
