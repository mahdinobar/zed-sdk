# ZED SDK - Geotracking for global scale localization on real-world map

## Sample Structure:
The samples provided with the geotracking API are organized as follows:

### Recording Sample
The Recording sample demonstrates how to record data from both a ZED camera and an external GNSS sensor. The recorded data is saved in an SVO file and a JSON file, respectively. This sample provides the necessary data for the Playback sample. 

### Playback Sample
The Playback sample shows how to use the geotracking API for global scale localization on a real-world map. It takes the data generated by the Recording sample and uses it to display geo-positions on a real-world map. 

### GeoTracking Sample
The GeoTracking sample demonstrates how to use the geotracking API for global scale localization on a real-world map using both the ZED camera and an external GNSS sensor. It displays the corrected positional tracking in the ZED reference frame on an OpenGL window. It also displays the geo-position on a real-world map on [ZED Hub](https://hub.stereolabs.com).

By utilizing these samples, developers can quickly and easily incorporate geotracking capabilities into their projects using the ZED SDK. 
Developers can take advantage of the benefits of the ZED SDK to quickly and easily incorporate geotracking capabilities into their projects.