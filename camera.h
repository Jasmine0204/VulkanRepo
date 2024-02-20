#pragma once

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// References: 
// https://stackoverflow.com/questions/71707566/opengl-first-person-realistic-keyboard-movement

class UserCamera {
public:
	// camera attributes
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;

	// euler angles
	float Yaw;
	float Pitch;

	// camera adjustments
	float MovementSpeed;
	float MouseSensitivity;

	// augments for culling
	float FOV;
	float AspectRatio;
	float NearPlane;
	float FarPlane;

	// add a flag for fixed camera
	bool IsFixed;

	UserCamera(
		glm::vec3 position = glm::vec3(1.0f, 0.0f, 1.0f),
		glm::vec3 up = glm::vec3(0.0f, 0.0, 1.0f),
		float yaw = -90.0f,
		float pitch = 0.0f,
		float fov = 45.0f,
		float aspectRatio = 1.7778f,
		float nearPlane = 0.1f,
		float farPlane = 100.0f,
		bool isFixed = false) :
		Front(glm::vec3(0.0f, 1.0f, 0.0f)),
		MovementSpeed(3.0f),
		MouseSensitivity(0.1f)
	{
		Position = position;
		WorldUp = up;
		Yaw = yaw;
		Pitch = pitch;
		FOV = fov;
		AspectRatio = aspectRatio;
		NearPlane = nearPlane;
		FarPlane = farPlane;
		IsFixed = isFixed;

		updateCameraVectors();
	}

	// functions that can change camera attr from outside
	void setFixedCamera(bool isFixed) {
		IsFixed = isFixed;
	}

	void setCameraTransform(glm::vec3 position, float yaw, float pitch) {
		Position = position;
		Yaw = yaw;
		Pitch = pitch;
		updateCameraVectors();
	}

	// calculate view matrix
	glm::mat4 GetViewMatrix() {
		return glm::lookAt(Position, Position + Front, Up);
	}

	glm::mat4 GetProjectionMatrix() const {
		return glm::perspective(glm::radians(FOV), AspectRatio, NearPlane, FarPlane);;
	}

	// process keyboard inputs
	void ProcessKeyboard(GLFWwindow* window, float deltaTime) {
			float velocity = MovementSpeed * deltaTime;
			if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
				Position += Front * velocity;
			if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
				Position -= Front * velocity;
			if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
				Position -= Right * velocity;
			if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
				Position += Right * velocity;
	}

	void ProcessMouseMovement(float xoffset, float yoffset) {
		xoffset *= MouseSensitivity;
		yoffset *= MouseSensitivity;

		Yaw += xoffset;
		Pitch += yoffset;


		if (Pitch > 89.0f)
		{
			Pitch = 89.0f;
		}
		if (Pitch < -89.0f)
		{
			Pitch = -89.0f;
		}

		updateCameraVectors();

	}


private:
	void updateCameraVectors() {

		// re-calculate front vector
		glm::vec3 front;
		front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
		front.y = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
		front.z = sin(glm::radians(Pitch));
		Front = glm::normalize(front);

		// re-caculate right vector
		Right = glm::normalize(glm::cross(Front, WorldUp));
		Up = glm::normalize(glm::cross(Right, Front));
	}
};