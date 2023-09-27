from visualization_msgs.msg import Marker, MarkerArray
import rospy
import numpy as  np

rospy.init_node('waypoint_publisher', anonymous=True)
waypoints = np.loadtxt('waypoints/waypoint_urban_station_v2.txt', ndmin=2)
num_wp = waypoints.shape[0]
print('waypoints:', waypoints)
print('num_wp:', num_wp)

waypoint_publisher = rospy.Publisher("/waypoints", MarkerArray)
marker_array = MarkerArray()
id = 0

for j in range(num_wp):
    marker = Marker()
    marker.id = id
    id = id + 1
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = ""
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = waypoints[j][0]
    marker.pose.position.y = waypoints[j][1]
    marker.pose.position.z = waypoints[j][2]
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker_array.markers.append(marker)

    # if j%2 == 0:
    text_marker = Marker()
    text_marker.id = id
    id = id + 1
    text_marker.header.frame_id = "world"
    text_marker.header.stamp = rospy.Time.now()
    text_marker.ns = ""
    text_marker.type = Marker.TEXT_VIEW_FACING
    text_marker.action = Marker.ADD
    if (j == 5):
        text_marker.pose.position.x = waypoints[j][0] - 1.0
        text_marker.pose.position.y = waypoints[j][1] - 0.25
        text_marker.pose.position.z = waypoints[j][2] + 1.25
    elif (j == 10):
        text_marker.pose.position.x = waypoints[j][0] + 1.5
        text_marker.pose.position.y = waypoints[j][1] + 0.25 
        text_marker.pose.position.z = waypoints[j][2] + 1.25
    elif (j == 4):
        text_marker.pose.position.x = waypoints[j][0] - 0.5
        text_marker.pose.position.y = waypoints[j][1]
        text_marker.pose.position.z = waypoints[j][2] + 1.25
    elif (j == 20):
        text_marker.pose.position.x = waypoints[j][0] + 0.5
        text_marker.pose.position.y = waypoints[j][1]
        text_marker.pose.position.z = waypoints[j][2] + 1.25        
    else:
        text_marker.pose.position.x = waypoints[j][0]
        text_marker.pose.position.y = waypoints[j][1]
        text_marker.pose.position.z = waypoints[j][2] + 1.25
    text_marker.pose.orientation.x = 0.0
    text_marker.pose.orientation.y = 0.0
    text_marker.pose.orientation.z = 0.0
    text_marker.pose.orientation.w = 1.0
    text_marker.scale.x = 2.0
    text_marker.scale.y = 2.0
    text_marker.scale.z = 2.0
    text_marker.color.a = 1.0
    text_marker.color.r = 1.0
    text_marker.color.g = 1.0
    text_marker.color.b = 0.0
    text_marker.frame_locked = False
    text_marker.text = str(j)
    marker_array.markers.append(text_marker)

rospy.sleep(1.0)
waypoint_publisher.publish(marker_array)


# while not rospy.is_shutdown():
#     rospy.sleep(1.0)
#     mesh_publisher.publish(msg)