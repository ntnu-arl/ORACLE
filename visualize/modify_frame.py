import rospy
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
import tf
from scipy.spatial.transform import Rotation as R

class FrameModifier:
    def __init__(self):
        rospy.init_node('traj_marker_modifier', anonymous=True)
        self.br = tf.TransformBroadcaster()
        self.odom_sub = rospy.Subscriber("/mavros/local_position/odom_in_map", Odometry, self.odom_callback)
        self.traj_sub = rospy.Subscriber("/trajectory", MarkerArray, self.traj_callback)
        self.traj_pub = rospy.Publisher("/trajectory_mod", MarkerArray)
    
    def odom_callback(self, msg):
        odom_quat = msg.pose.pose.orientation
        r_robot = R.from_quat([odom_quat.x, odom_quat.y, odom_quat.z, odom_quat.w])
        robot_euler_angles = r_robot.as_euler('xyz', degrees=False)
        # print('robot_euler_angles:', robot_euler_angles * 180 / 3.14159)
        self.br.sendTransform((0.0, 0.0, 0.0),
                        tf.transformations.quaternion_from_euler(-robot_euler_angles[0], -robot_euler_angles[1], 0),
                        msg.header.stamp,
                        "vehicle",
                        "state")        

    def traj_callback(self, msg):
        # msg_mod = MarkerArray()
        for i in range(len(msg.markers)):
            msg.markers[i].header.frame_id = 'vehicle' 
            # marker = msg.markers[i]
            # if marker.color.r == 0.0 and marker.color.g == 1.0 and marker.color.b == 0.0: # safe action seq
            #     marker.color.a = 1.0
            #     marker.scale.x = 0.1
            #     marker.scale.y = 0.1
            #     marker.scale.z = 0.1
            # elif marker.color.r == 1.0 and marker.color.g == 1.0 and marker.color.b == 0.0: # unsafe action seq
            #     marker.color.g = 0.647 # orange
            #     marker.color.a = 1.0
            #     marker.scale.x = 0.1
            #     marker.scale.y = 0.1
            #     marker.scale.z = 0.1
            # elif marker.color.r == 0.0 and marker.color.g == 0.0 and marker.color.b == 1.0: # best action seq
            #     marker.color.a = 1.0
            #     marker.scale.x = 0.3
            #     marker.scale.y = 0.3
            #     marker.scale.z = 0.3            
            # elif marker.color.r == 1.0 and marker.color.g == 0.0 and marker.color.b == 0.0: # worst action seq
            #     marker.color.a = 1.0
            #     marker.scale.x = 0.1
            #     marker.scale.y = 0.1
            #     marker.scale.z = 0.1
            # msg_mod.markers.append(marker)            

        self.traj_pub.publish(msg)

if __name__ == '__main__':
    rospy.loginfo('Ready')
    obj = FrameModifier()
    rospy.spin()