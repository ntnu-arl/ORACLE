from visualization_msgs.msg import Marker
import rospy

rospy.init_node('env_mesh_publisher', anonymous=True)

# mesh_publisher = rospy.Publisher("/env_mesh", Marker)
mesh_publisher_obj1 = rospy.Publisher("/obj1", Marker)
mesh_publisher_obj2 = rospy.Publisher("/obj2", Marker)
mesh_publisher_obj3 = rospy.Publisher("/obj3", Marker)
mesh_publisher_obj4 = rospy.Publisher("/obj4", Marker)
mesh_publisher_obj5 = rospy.Publisher("/obj5", Marker)
mesh_publisher_obj6 = rospy.Publisher("/obj6", Marker)

# env
# msg = Marker()
# msg.header.frame_id = "world"
# msg.header.stamp = rospy.Time.now()
# msg.ns = ""
# msg.id = 0
# msg.type = Marker.MESH_RESOURCE
# msg.action = Marker.ADD
# msg.pose.position.x = 0
# msg.pose.position.y = 0
# msg.pose.position.z = 6.8
# msg.pose.orientation.x = 0.0
# msg.pose.orientation.y = 0.0
# msg.pose.orientation.z = 0.7068252
# msg.pose.orientation.w = 0.7073883
# msg.scale.x = 1.0
# msg.scale.y = 1.0
# msg.scale.z = 1.0
# msg.mesh_resource = "file:///home/huan/subway_meshes/Urban_Station/meshes/station.dae"
# msg.mesh_use_embedded_materials = True

# obj1
msg_obj1 = Marker()
msg_obj1.header.frame_id = "world"
msg_obj1.header.stamp = rospy.Time.now()
msg_obj1.ns = ""
msg_obj1.id = 0
msg_obj1.type = Marker.MESH_RESOURCE
msg_obj1.action = Marker.ADD
msg_obj1.pose.position.x = 2.05015
msg_obj1.pose.position.y = -4.86573
msg_obj1.pose.position.z = 1.98398
msg_obj1.pose.orientation.x = 0.0
msg_obj1.pose.orientation.y = 0.0
msg_obj1.pose.orientation.z = -0.3996225 # -0.82221
msg_obj1.pose.orientation.w = 0.9166798
msg_obj1.scale.x = 3
msg_obj1.scale.y = 3
msg_obj1.scale.z = 3
msg_obj1.mesh_resource = "file:///home/huan/subway_meshes/JanSport_Backpack_Red/meshes/backpack.dae"
msg_obj1.mesh_use_embedded_materials = True

# obj2
msg_obj2 = Marker()
msg_obj2.header.frame_id = "world"
msg_obj2.header.stamp = rospy.Time.now()
msg_obj2.ns = ""
msg_obj2.id = 0
msg_obj2.type = Marker.MESH_RESOURCE
msg_obj2.action = Marker.ADD
msg_obj2.pose.position.x = -13.5328
msg_obj2.pose.position.y = 10.6484
msg_obj2.pose.position.z = 8.92571
msg_obj2.pose.orientation.x = 0.0 # 0.6532814, 0.270598, 0.270598, 0.6532816
msg_obj2.pose.orientation.y = 0.0
msg_obj2.pose.orientation.z = 0.0
msg_obj2.pose.orientation.w = 1.0
msg_obj2.scale.x = 3
msg_obj2.scale.y = 3
msg_obj2.scale.z = 3
msg_obj2.mesh_resource = "file:///home/huan/subway_meshes/JanSport_Backpack_Red/meshes/backpack.dae"
msg_obj2.mesh_use_embedded_materials = True

# obj3
msg_obj3 = Marker()
msg_obj3.header.frame_id = "world"
msg_obj3.header.stamp = rospy.Time.now()
msg_obj3.ns = ""
msg_obj3.id = 0
msg_obj3.type = Marker.MESH_RESOURCE
msg_obj3.action = Marker.ADD
msg_obj3.pose.position.x = -6.28176
msg_obj3.pose.position.y = -4.04658
msg_obj3.pose.position.z = 8.48498
msg_obj3.pose.orientation.x = 0.0 # 0.6532814, 0.270598, 0.270598, 0.6532816
msg_obj3.pose.orientation.y = 0.0
msg_obj3.pose.orientation.z = -0.8328738 # -1.96856
msg_obj3.pose.orientation.w = 0.5534629
msg_obj3.scale.x = 3
msg_obj3.scale.y = 3
msg_obj3.scale.z = 3
msg_obj3.mesh_resource = "file:///home/huan/subway_meshes/JanSport_Backpack_Red/meshes/backpack.dae"
msg_obj3.mesh_use_embedded_materials = True

# obj4
msg_obj4 = Marker()
msg_obj4.header.frame_id = "world"
msg_obj4.header.stamp = rospy.Time.now()
msg_obj4.ns = ""
msg_obj4.id = 0
msg_obj4.type = Marker.MESH_RESOURCE
msg_obj4.action = Marker.ADD
msg_obj4.pose.position.x = 24.9753
msg_obj4.pose.position.y = 2.72513
msg_obj4.pose.position.z = 8.16189
msg_obj4.pose.orientation.x = 0.0 # 0.6532814, 0.270598, 0.270598, 0.6532816
msg_obj4.pose.orientation.y = 0.0
msg_obj4.pose.orientation.z = -0.5624244 # -1.19463
msg_obj4.pose.orientation.w = 0.8268487
msg_obj4.scale.x = 3
msg_obj4.scale.y = 3
msg_obj4.scale.z = 3
msg_obj4.mesh_resource = "file:///home/huan/subway_meshes/JanSport_Backpack_Red/meshes/backpack.dae"
msg_obj4.mesh_use_embedded_materials = True

# obj5
msg_obj5 = Marker()
msg_obj5.header.frame_id = "world"
msg_obj5.header.stamp = rospy.Time.now()
msg_obj5.ns = ""
msg_obj5.id = 0
msg_obj5.type = Marker.MESH_RESOURCE
msg_obj5.action = Marker.ADD
msg_obj5.pose.position.x = -2.32309
msg_obj5.pose.position.y = 1.67989
msg_obj5.pose.position.z = 2.5502
msg_obj5.pose.orientation.x = 0.0 # 0.6532814, 0.270598, 0.270598, 0.6532816
msg_obj5.pose.orientation.y = 0.0
msg_obj5.pose.orientation.z = 0.3228834
msg_obj5.pose.orientation.w = 0.9464387 # 0.657549
msg_obj5.scale.x = 3
msg_obj5.scale.y = 3
msg_obj5.scale.z = 3
msg_obj5.mesh_resource = "file:///home/huan/subway_meshes/JanSport_Backpack_Red/meshes/backpack.dae"
msg_obj5.mesh_use_embedded_materials = True

# obj6
msg_obj6 = Marker()
msg_obj6.header.frame_id = "world"
msg_obj6.header.stamp = rospy.Time.now()
msg_obj6.ns = ""
msg_obj6.id = 0
msg_obj6.type = Marker.MESH_RESOURCE
msg_obj6.action = Marker.ADD
msg_obj6.pose.position.x = 18.3842
msg_obj6.pose.position.y = -2.63902
msg_obj6.pose.position.z = 1.92866
msg_obj6.pose.orientation.x = 0.0 # 0.6532814, 0.270598, 0.270598, 0.6532816
msg_obj6.pose.orientation.y = 0.0
msg_obj6.pose.orientation.z = 0.2811507 # 0.569986
msg_obj6.pose.orientation.w = 0.9596636
msg_obj6.scale.x = 3
msg_obj6.scale.y = 3
msg_obj6.scale.z = 3
msg_obj6.mesh_resource = "file:///home/huan/subway_meshes/JanSport_Backpack_Red/meshes/backpack.dae"
msg_obj6.mesh_use_embedded_materials = True

# rospy.sleep(1.0)
# mesh_publisher.publish(msg)
rospy.sleep(1.0)
mesh_publisher_obj1.publish(msg_obj1)
rospy.sleep(1.0)
mesh_publisher_obj2.publish(msg_obj2)
rospy.sleep(1.0)
mesh_publisher_obj3.publish(msg_obj3)
rospy.sleep(1.0)
mesh_publisher_obj4.publish(msg_obj4)
rospy.sleep(1.0)
mesh_publisher_obj5.publish(msg_obj5)
rospy.sleep(1.0)
mesh_publisher_obj6.publish(msg_obj6)

# while not rospy.is_shutdown():
#     rospy.sleep(1.0)
#     mesh_publisher.publish(msg)
#     rospy.sleep(1.0)
#     mesh_publisher_obj1.publish(msg_obj1)
#     rospy.sleep(1.0)
#     mesh_publisher_obj2.publish(msg_obj2)
#     rospy.sleep(1.0)
#     mesh_publisher_obj3.publish(msg_obj3)
#     rospy.sleep(1.0)
#     mesh_publisher_obj4.publish(msg_obj4)