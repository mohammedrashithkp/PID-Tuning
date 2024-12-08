#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

import time


class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(None, None), sample_time=0.01, filter_coefficient=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output, self.max_output = output_limits
        self.sample_time = sample_time
        self.filter_coefficient = filter_coefficient

        self.integral = 0
        self.previous_error = 0
        self.previous_time = None
        self.filtered_derivative = 0

    def update(self, setpoint, measurement, current_time, use_lpf=True):
        error = setpoint - measurement

        if self.previous_time is None:
            self.previous_time = current_time
            return 0
        delta_time = current_time - self.previous_time
        if delta_time < self.sample_time:
            return 0

        p_term = self.kp * error

        self.integral += error * delta_time
        i_term = self.ki * self.integral

        raw_derivative = (error - self.previous_error) / delta_time if delta_time > 0 else 0

        # Noise filtering
        if use_lpf:
            self.filtered_derivative = (
                self.filter_coefficient * raw_derivative
                + (1 - self.filter_coefficient) * self.filtered_derivative
            )
        else:
            self.filtered_derivative = raw_derivative

        d_term = self.kd * self.filtered_derivative

        # Calculate total output
        output = p_term + i_term + d_term

        # Apply anti-windup clamping
        if self.min_output is not None and self.max_output is not None:
            if output > self.max_output:
                output = self.max_output
                self.integral -= error * delta_time
            elif output < self.min_output:
                output = self.min_output
                self.integral -= error * delta_time

        self.previous_error = error
        self.previous_time = current_time

        return output


class DiffDriveController(Node):
    def __init__(self, pid_params, tuning_method, noise_filter):
        super().__init__('diff_drive_controller')

        self.kp, self.ki, self.kd = pid_params
        self.noise_filter = noise_filter
        self.tuning_method = tuning_method
        self.pid = PIDController(
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            output_limits=(-255, 255),
            filter_coefficient=0.1,
        )

        self.cmd_vel_sub = self.create_subscription(Twist, "cmd_vel", self.recv_cmd_vel, 1)
        self.left_pub = self.create_publisher(Float32, "left_wheel/control_effort", 1)
        self.right_pub = self.create_publisher(Float32, "right_wheel/control_effort", 1)

        self.dynamic_pid_tuning()

    def recv_cmd_vel(self, msg: Twist):
        v = msg.linear.x
        w = msg.angular.z

        v_left = -(v + (w * self.wheel_separation / 2))
        v_right = -(v - (w * self.wheel_separation / 2))

        current_time = self.get_clock().now().nanoseconds / 1e9
        pwm_left = self.pid.update(v_left, 0, current_time, use_lpf=(self.noise_filter == "lpf"))
        pwm_right = self.pid.update(v_right, 0, current_time, use_lpf=(self.noise_filter == "lpf"))

        self.left_pub.publish(Float32(data=pwm_left))
        self.right_pub.publish(Float32(data=pwm_right))

        self.get_logger().info(f"PWM Left: {pwm_left:.2f}, PWM Right: {pwm_right:.2f}")

    def dynamic_pid_tuning(self):
        Ku = 10.0
        Tu = 2.0
        Kp = 1.0
        T = 2.0
        L = 1.0

        if self.tuning_method == "ziegler-nichols":
            self.kp = 0.6 * Ku
            self.ki = 2 * self.kp / Tu
            self.kd = self.kp * Tu / 8
        elif self.tuning_method == "cohen-coon":
            self.kp = (1.35 / Kp) * (T / L + 0.185)
            self.ki = 2.5 * L / (T + 0.185 * L)
            self.kd = 0.37 * L * T / (T + 0.185 * L)

        self.pid.kp = self.kp
        self.pid.ki = self.ki
        self.pid.kd = self.kd


def show_menu(console):
    table = Table(title="PID Controller Menu")
    table.add_column("Option", justify="center")
    table.add_column("Description", justify="left")

    table.add_row("1", "Set PID parameters")
    table.add_row("2", "Select Noise Filter (lpf/none)")
    table.add_row("3", "Select Tuning Method (Cohen-Coon / Ziegler-Nichols)")
    table.add_row("4", "Run Simulation")
    table.add_row("5", "Exit")

    console.print(table)


def main():
    rclpy.init()

    console = Console()

    # Default values
    pid_params = (1.0, 0.0, 0.0)
    tuning_method = "cohen-coon"
    noise_filter = "lpf"

    while True:
        show_menu(console)
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"], default="5")

        if choice == "1":
            kp = float(Prompt.ask("Enter Kp", default="1.0"))
            ki = float(Prompt.ask("Enter Ki", default="0.0"))
            kd = float(Prompt.ask("Enter Kd", default="0.0"))
            pid_params = (kp, ki, kd)
            console.print(f"Updated PID Parameters: Kp={kp}, Ki={ki}, Kd={kd}")

        elif choice == "2":
            noise_filter = Prompt.ask("Select Noise Filter (lpf or none)", choices=["lpf", "none"], default="lpf")
            console.print(f"Selected Noise Filter: {noise_filter}")

        elif choice == "3":
            tuning_method = Prompt.ask(
                "Select Tuning Method (Cohen-Coon or Ziegler-Nichols)",
                choices=["cohen-coon", "ziegler-nichols"],
                default="cohen-coon",
            )
            console.print(f"Selected Tuning Method: {tuning_method}")

        elif choice == "4":
            controller = DiffDriveController(pid_params, tuning_method, noise_filter)
            rclpy.spin(controller)

        elif choice == "5":
            console.print("Exiting...")
            break

    rclpy.shutdown()


if __name__ == "__main__":
    main()

