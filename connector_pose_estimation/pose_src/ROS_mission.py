import os
import time
import subprocess
import signal

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

class TriggerManager(Node):
    def __init__(self):
        super().__init__('connector_missionor')
        self.declare_parameter('poseEstmationA_launch_args', ['mesh_filename:=ConnectorA.obj'])
        self.declare_parameter('poseEstmationB_launch_args', ['mesh_filename:=ConnectorA.obj'])
        
        self.trigger_topic = '/poseEstimation/trigger'
        self.triggerEnd_topic = '/poseEstimation/triggerEnd'

        self.create_subscription(String, self.trigger_topic, self.trigger_callback, 10)
        self.create_subscription(Bool, self.triggerEnd_topic, self.trigger_end_callback, 10)

        self.desired_camera = None
        self.trigger_end = False
        self.active_process = None

        self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("[POSE ESTIMATOR] 커넥터 Pose Estimation 센서 트리거 대기")

    def trigger_callback(self, msg: String):
        self.get_logger().info(f"[POSE ESTIMATOR] 커넥터 Pose Estimation 옵션: {msg.data}")
        if msg.data in ['Alice', 'Bob']:
            self.desired_camera = msg.data
            self.trigger_end = False
        else:
            self.get_logger().warn(f"[POSE ESTIMATOR] 커넥터 Pose Estimation 트리거 오류: {msg.data}")

    def trigger_end_callback(self, msg: Bool):
        if msg.data:
            self.trigger_end = True

    def timer_callback(self):
        # 프로세스가 실행 중이지 않고, 원하는 trigger가 수신되었다면 실행
        if self.active_process is None and self.desired_camera is not None:
            if self.desired_camera == 'Alice':
                time.sleep(3)
                self.get_logger().info("[POSE ESTIMATOR] A_Side 커넥터 Pose Estimation 시작")
                launch_args = self.get_parameter('poseEstmationA_launch_args').get_parameter_value().string_array_value
                self.get_logger().info(f"[POSE ESTIMATOR] A_Side 커넥터 Pose Estimation 옵션: {launch_args}")
                self.active_process = subprocess.Popen(
                    ['ros2', 'launch', 'connector_pose_estimation', 'pose_estimator_launch.py'] + launch_args,
                    preexec_fn=os.setsid
                )
            elif self.desired_camera == 'Bob':
                time.sleep(3)
                self.get_logger().info("[POSE ESTIMATOR] L515 DPT Sensor B (Bob) 시작")
                launch_args = self.get_parameter('poseEstmationB_launch_args').get_parameter_value().string_array_value
                self.get_logger().info(f"[POSE ESTIMATOR] B_Side 커넥터 Pose Estimation 옵션: {launch_args}")
                self.active_process = subprocess.Popen(
                    ['ros2', 'launch', 'connector_pose_estimation', 'pose_estimator_launch.py'] + launch_args,
                    preexec_fn=os.setsid
                )

        if self.active_process is not None and self.trigger_end:
            self.get_logger().warn("[POSE ESTIMATOR] 커넥터 Pose Estimation 종료")
            try:
                os.killpg(self.active_process.pid, signal.SIGINT)
                time.sleep(2)
                if self.active_process.poll() is None:
                    os.killpg(self.active_process.pid, signal.SIGKILL)
                self.active_process.wait()
            except Exception as e:
                self.get_logger().error(f"[POSE ESTIMATOR] 인터럽트: {e}")
            finally:
                self.active_process = None
                self.desired_camera = None
                self.trigger_end = False

def main(args=None):
    rclpy.init(args=args)
    node = TriggerManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
