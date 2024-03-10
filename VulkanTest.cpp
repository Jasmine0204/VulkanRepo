#define _USE_MATH_DEFINES
#include <cmath>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#define GLM_FORCE_RADIANS
// need to configure the persp proj matrix to use the Vulkan range of 0.0 to 1.0
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <chrono> // precise timekeeping
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <variant>
#include <set>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <array>


#include "parser.hpp"
#include "camera.h"
#include "event.h"
#include "lambertian.h"

// include image loader
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


#define STB_IMAGE_WRITE_IMPLEMENTATION
#define _CRT_SECURE_NO_WARNINGS
#include <stb_image_write.h>

// Written by Jasmine Chen, andrew id: zitongc
// For 15-672 Realtime Rendering Graphics during Spring 2024, taught by Jim McCann
// References: 
// A1:
// https://docs.vulkan.org/tutorial/latest/00_Introduction.html basically everything in initVulkan()
// https://easyvulkan.github.io/ some supplement materials
// https://stackoverflow.com/questions/5782658/extracting-yaw-from-a-quaternion convert quat to yaw and pitch
// https://raytracing.github.io/books/RayTracingTheNextWeek.html#boundingvolumehierarchies/axis-alignedboundingboxes(aabbs) the inspiration of AABB bounding box
// https://www.glfw.org/docs/latest/quick.html GLFW documentations
// https://cplusplus.com/articles/DEN36Up4/ parse command line options
// A2:
// https://www.radiance-online.org/cgi-bin/viewcvs.cgi/ray/src/common/color.c?revision=2.33&view=markup#l188 convert rgbe to rgb
// https://github.com/ixchow/15-466-ibl/blob/master/rgbe.hpp rgbe -> rgb
// https://learnopengl.com/Advanced-Lighting/HDR hdr & tone-mapping
// https://64.github.io/tonemapping/ tone-mapping
// https://gist.github.com/Pikachuxxxx/136940d6d0d64074aba51246f514bd26 tone-mapping in fragment shader
// https://en.wikipedia.org/wiki/Relative_luminance adjust vibrance after tone-mapping
// https://architextures.org/textures free educational use textures

uint32_t WIDTH = 1280;
uint32_t HEIGHT = 720;
const int MAX_FRAME_IN_FLIGHT = 2;
uint32_t currentFrame = 0;
std::string eventFilePath = "./model/events.txt";
bool headless = false;
bool lambertian = false;
std::string inputFile = "./model/ox_bridge_morning.png";
std::string outputFile = "./model/sky_diffuse.png";


// store all supportive command line options
struct CommandLineOptions {
	bool headless = false;
	std::string drawingSize;
	std::string sceneFile = "./model/materials.s72";
	std::string eventFile = "./model/events.txt";
	bool lambertian = false;
	std::string inputFile = "./model/ox_bridge_morning.png";
	std::string outputFile = "./model/sky_diffuse.png";
};


// get sceneGraph data
SceneGraph sceneGraph;
const std::vector<Mesh>& meshes = sceneGraph.getMeshes();
const std::vector<Node>& nodes = sceneGraph.getNodes();
const std::vector<Camera>& cameras = sceneGraph.getCameras();
const std::vector<AnimationClip>& clips = sceneGraph.getClips();
std::vector<Material>& materials = sceneGraph.getMaterials();
const Scene& scene = sceneGraph.getScene();
const Environment& environment = sceneGraph.getEnvironment();


// read b72 file
std::vector<char> readBinaryFile(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary | std::ios::ate);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file: " + filename);
	}

	size_t fileSize = (size_t)file.tellg();
	std::vector<char> buffer;

	file.seekg(0);
	buffer.resize(fileSize);
	file.read(buffer.data(), fileSize);
	file.close();

	return buffer;
}

// parse vertices (& vertices with simple material)
std::vector<Vertex> parseVertices(const std::string fileName, const Mesh& mesh) {
	std::vector<char> verticesBuffer = readBinaryFile("./model/" + fileName);

	std::vector<Vertex> vertices;

	size_t numVertices = mesh.count;

	for (size_t i = 0; i < numVertices; ++i) {
		Vertex vertex;

		// parse position
		size_t positionOffset = i * 52;
		if (positionOffset + sizeof(vertex.pos) > verticesBuffer.size()) {
			throw std::runtime_error("Position data exceeds buffer limits");
		}
		memcpy(&vertex.pos, verticesBuffer.data() + positionOffset, sizeof(vertex.pos));

		// parse normal
		size_t normalOffset = i * 52 + 12;
		if (normalOffset + sizeof(vertex.normal) > verticesBuffer.size()) {
			throw std::runtime_error("Normal data exceeds buffer limits");
		}
		memcpy(&vertex.normal, verticesBuffer.data() + normalOffset, sizeof(vertex.normal));

		// parse tangent
		size_t tangentOffset = i * 52 + 24;
		if (tangentOffset + sizeof(vertex.tangent) > verticesBuffer.size()) {
			throw std::runtime_error("tangent data exceeds buffer limits");
		}
		memcpy(&vertex.tangent, verticesBuffer.data() + tangentOffset, sizeof(vertex.tangent));
		//std::cout << vertex.tangent.x << " " << vertex.tangent.y << " " << vertex.tangent.z << " " << vertex.tangent.w << "\n";
		// parse texcoord
		size_t texOffset = i * 52 + 40;
		if (texOffset + sizeof(vertex.texCoord) > verticesBuffer.size()) {
			throw std::runtime_error("texcoord data exceeds buffer limits");
		}
		memcpy(&vertex.texCoord, verticesBuffer.data() + texOffset, sizeof(vertex.texCoord));

		// parse color
		size_t colorOffset = i * 52 + 48;

		uint32_t color;
		if (colorOffset + sizeof(color) > verticesBuffer.size()) {
			throw std::runtime_error("Color data exceeds buffer limits");
		}

		memcpy(&vertex.color.r, verticesBuffer.data() + colorOffset, sizeof(uint32_t)); // R
		memcpy(&vertex.color.g, verticesBuffer.data() + colorOffset + 4, sizeof(uint32_t)); // G
		memcpy(&vertex.color.b, verticesBuffer.data() + colorOffset + 8, sizeof(uint32_t)); // B
		memcpy(&vertex.color.a, verticesBuffer.data() + colorOffset + 12, sizeof(uint32_t)); // A

		vertices.push_back(vertex);
	}
	return vertices;
}

std::vector<SimpleVertex> parseSimpleVertices(const std::string fileName, const Mesh& mesh) {
	std::vector<char> verticesBuffer = readBinaryFile("./model/" + fileName);

	std::vector<SimpleVertex> simpleVertices;

	size_t numVertices = mesh.count;

	for (size_t i = 0; i < numVertices; ++i) {
		SimpleVertex simpleVertex;

		// parse position
		size_t positionOffset = i * 28;
		if (positionOffset + sizeof(simpleVertex.pos) > verticesBuffer.size()) {
			throw std::runtime_error("Position data exceeds buffer limits");
		}
		memcpy(&simpleVertex.pos, verticesBuffer.data() + positionOffset, sizeof(simpleVertex.pos));

		// parse normal
		size_t normalOffset = i * 28 + 12;
		if (normalOffset + sizeof(simpleVertex.normal) > verticesBuffer.size()) {
			throw std::runtime_error("Normal data exceeds buffer limits");
		}
		memcpy(&simpleVertex.normal, verticesBuffer.data() + normalOffset, sizeof(simpleVertex.normal));

		// parse color
		uint32_t color;
		size_t colorOffset = i * 28 + 24;
		if (colorOffset + sizeof(color) > verticesBuffer.size()) {
			throw std::runtime_error("Color data exceeds buffer limits");
		}

		memcpy(&simpleVertex.color.r, verticesBuffer.data() + colorOffset, sizeof(uint32_t)); // R
		memcpy(&simpleVertex.color.g, verticesBuffer.data() + colorOffset + 4, sizeof(uint32_t)); // G
		memcpy(&simpleVertex.color.b, verticesBuffer.data() + colorOffset + 8, sizeof(uint32_t)); // B
		memcpy(&simpleVertex.color.a, verticesBuffer.data() + colorOffset + 12, sizeof(uint32_t)); // A

		simpleVertices.push_back(simpleVertex);
	}

	return simpleVertices;
}

const std::vector<uint32_t> indices = {
	0,1,2,2,3,0,
	4,5,6,6,7,4
};

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// tell Vulkan about the UBO descriptor
struct UniformBufferObject {
	glm::mat4 view;
	glm::mat4 projection;
	glm::vec4 cameraPos;
};

// store push constants struct
struct PushConstants {
	glm::mat4 model;
	glm::vec4 albedoColor;
	float roughness;
	float metalness;
	int materialType;
	// 0 for simple material, 1/2/3/4 for complex material
};

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		if (!headless) {
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
		else {
			return graphicsFamily.has_value();
		}
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};


// Bouding box
// I choose AABB.
// define struct
struct BoundingBox {
	glm::vec3 minPoint;
	glm::vec3 maxPoint;

	// constructor
	BoundingBox() : minPoint(glm::vec3(FLT_MAX)), maxPoint(glm::vec3(-FLT_MAX)) {}

	// calculate AABB
	BoundingBox calculateBoundingBox(const Node& node, const SceneGraph& sceneGraph, const glm::mat4& modelMatrix) {
		BoundingBox box;

		int containerIndex = sceneGraph.meshIndexMap.at(node.mesh);
		const Mesh& mesh = sceneGraph.meshes.at(containerIndex);
		Attribute attr = mesh.getAttribute("POSITION");
		bool isSimple;
		int stride = attr.stride;

		if (stride == 28) {
			isSimple = true;
		}
		else {
			isSimple = false;
		}


		if (!isSimple) {
			std::vector<Vertex> vertices = parseVertices(attr.src, mesh);
			for (const auto& point : vertices) {
				glm::vec4 worldPoint = modelMatrix * glm::vec4(point.pos, 1.0f); // traverse vertices to world
				glm::vec3 transformedPoint = glm::vec3(worldPoint.x, worldPoint.y, worldPoint.z);
				box.minPoint = glm::min(box.minPoint, transformedPoint);
				box.maxPoint = glm::max(box.maxPoint, transformedPoint);
			}
			return box;
		}
		else {
			std::vector<SimpleVertex> simpleVertices = parseSimpleVertices(attr.src, mesh);
			for (const auto& point : simpleVertices) {
				glm::vec4 worldPoint = modelMatrix * glm::vec4(point.pos, 1.0f); // traverse vertices to world
				glm::vec3 transformedPoint = glm::vec3(worldPoint.x, worldPoint.y, worldPoint.z);
				box.minPoint = glm::min(box.minPoint, transformedPoint);
				box.maxPoint = glm::max(box.maxPoint, transformedPoint);
			}
			return box;
		}
	}
};


// frustum - view & projection matrix
bool isInsideFrustum(const BoundingBox& box, const glm::mat4& vpMatrix) {
	std::array<glm::vec4, 6> frustumPlanes;

	// get planes from view project matrices
	// right
	frustumPlanes[0] = glm::vec4(vpMatrix[0][3] - vpMatrix[0][0], vpMatrix[1][3] - vpMatrix[1][0], vpMatrix[2][3] - vpMatrix[2][0], vpMatrix[3][3] - vpMatrix[3][0]);
	// left
	frustumPlanes[1] = glm::vec4(vpMatrix[0][3] + vpMatrix[0][0], vpMatrix[1][3] + vpMatrix[1][0], vpMatrix[2][3] + vpMatrix[2][0], vpMatrix[3][3] + vpMatrix[3][0]);
	// top
	frustumPlanes[2] = glm::vec4(vpMatrix[0][3] - vpMatrix[0][1], vpMatrix[1][3] - vpMatrix[1][1], vpMatrix[2][3] - vpMatrix[2][1], vpMatrix[3][3] - vpMatrix[3][1]);
	// bottom
	frustumPlanes[3] = glm::vec4(vpMatrix[0][3] + vpMatrix[0][1], vpMatrix[1][3] + vpMatrix[1][1], vpMatrix[2][3] + vpMatrix[2][1], vpMatrix[3][3] + vpMatrix[3][1]);
	// near
	frustumPlanes[4] = glm::vec4(vpMatrix[0][3] + vpMatrix[0][2], vpMatrix[1][3] + vpMatrix[1][2], vpMatrix[2][3] + vpMatrix[2][2], vpMatrix[3][3] + vpMatrix[3][2]);
	// far
	frustumPlanes[5] = glm::vec4(vpMatrix[0][3] - vpMatrix[0][2], vpMatrix[1][3] - vpMatrix[1][2], vpMatrix[2][3] - vpMatrix[2][2], vpMatrix[3][3] - vpMatrix[3][2]);

	// normalize the plane
	for (auto& plane : frustumPlanes) {
		plane /= glm::length(glm::vec3(plane));
	}

	// detect if every corner is outside the frustum
	for (const auto& plane : frustumPlanes) {
		// if eight corners are on the negative direction of the planes
		if (glm::dot(glm::vec3(plane), glm::vec3(box.minPoint.x, box.minPoint.y, box.minPoint.z)) + plane.w > 0) continue;
		if (glm::dot(glm::vec3(plane), glm::vec3(box.maxPoint.x, box.minPoint.y, box.minPoint.z)) + plane.w > 0) continue;
		if (glm::dot(glm::vec3(plane), glm::vec3(box.minPoint.x, box.maxPoint.y, box.minPoint.z)) + plane.w > 0) continue;
		if (glm::dot(glm::vec3(plane), glm::vec3(box.maxPoint.x, box.maxPoint.y, box.minPoint.z)) + plane.w > 0) continue;
		if (glm::dot(glm::vec3(plane), glm::vec3(box.minPoint.x, box.minPoint.y, box.maxPoint.z)) + plane.w > 0) continue;
		if (glm::dot(glm::vec3(plane), glm::vec3(box.maxPoint.x, box.minPoint.y, box.maxPoint.z)) + plane.w > 0) continue;
		if (glm::dot(glm::vec3(plane), glm::vec3(box.minPoint.x, box.maxPoint.y, box.maxPoint.z)) + plane.w > 0) continue;
		if (glm::dot(glm::vec3(plane), glm::vec3(box.maxPoint.x, box.maxPoint.y, box.maxPoint.z)) + plane.w > 0) continue;

		return false;
	}
}




class FirstTriangle {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

	void runHeadless() {
		isDebugCameraActive = false;
		isRenderCameraActive = true;
		isUserCameraActive = false;
		initVulkanHeadless();
		auto events = parseEventFile(eventFilePath);
		handleEvents(events);
		cleanupHeadless();
	}



private:
	// classmembers

	// inits
	GLFWwindow* window;
	VkInstance instance;

	// physical and logical devices
	VkDevice device;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkPhysicalDeviceFeatures deviceFeatures{};

	// rendering queues
	VkQueue graphicsQueue;
	VkSurfaceKHR surface;
	VkQueue presentQueue;

	// swapchain relates
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;

	// headless mode image relates
	VkFormat offscreenImageFormat = VK_FORMAT_R8G8B8A8_UNORM;
	VkExtent2D offscreenExtent = { 1280, 720 };
	VkFramebuffer offscreenFramebuffer;
	VkImage offscreenImage;
	VkDeviceMemory offscreenImageMemory;
	VkImageView offscreenImageView;
	std::vector<VkCommandBuffer> headlessCommandBuffers;

	// descriptors and pipeplines
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkRenderPass swapChainRenderPass;
	VkRenderPass offscreenRenderPass;
	VkPipeline graphicsPipeline;
	VkPipeline simpleGraphicsPipeline;

	// descriptors
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	// framebuffer
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> swapChainCommandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;

	// vertex and index buffers
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;

	// ubos
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;

	// depth image
	VkImage depthImage;
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	// user camera
	UserCamera userCamera;

	// camera time setting
	std::chrono::steady_clock::time_point lastTime;
	float deltaTime;

	// debug camera settings
	UserCamera debugCamera;
	glm::vec3 debugCameraPosition = glm::vec3(3.0f, -1.0f, 5.0f);
	float debugCameraYaw = 180;
	float debugCameraPitch = -60.0;

	// render camera settings
	UserCamera renderCamera;
	glm::vec3 renderCameraPosition;
	float renderCameraYaw;
	float renderCameraPitch;

	// check camera state
	bool isDebugCameraActive = false;
	bool isRenderCameraActive = false;
	bool isUserCameraActive = true;

	// bounding box~~~
	const BoundingBox box;

	// resizing
	bool frameBufferResized = false;

	// animation time
	bool isPlaying = true;
	std::chrono::steady_clock::time_point startTime;
	std::chrono::steady_clock::time_point pauseTime;
	double animTime;
	float playbackRate = 1.0f;
	long long animationStartTime = 0;

	// push constants
	PushConstants pushConstants;

	// texture sampler
	VkSampler textureSampler;

	// environment cubemap
	Cubemap cubemap;
	VkImage cubemapImage;
	VkDeviceMemory cubemapImageMemory;
	VkImageView cubemapImageView;

	// lambertian cubemap
	Cubemap lambertianCubemap;
	VkImage lambertianCubemapImage;
	VkDeviceMemory lambertianCubemapImageMemory;
	VkImageView lambertianCubemapImageView;

	// default pixel
	VkImage defaultImage;
	VkDeviceMemory defaultImageMemory;
	VkImageView defaultImageView;

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);

		glfwSetCursorPosCallback(window, mouseCallback);
		glfwSetKeyCallback(window, keyCallback);

		//catch the cursor and hide it 
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	}

	static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
		auto app = reinterpret_cast<FirstTriangle*>(glfwGetWindowUserPointer(window));
		if (app) {
			app->keyControl(window, key, scancode, action, mods);
		}
	}

	// key controls
	void keyControl(GLFWwindow* window, int key, int scancode, int action, int mods) {
		if (action == GLFW_PRESS) {
			switch (key) {
				// space to pause animation
			case GLFW_KEY_SPACE:
				isPlaying = !isPlaying;
				if (isPlaying) {
					// keep playing and record time
					auto now = std::chrono::high_resolution_clock::now();
					startTime += now - pauseTime; // change start time 
				}
				else {
					pauseTime = std::chrono::high_resolution_clock::now();
				}
				break;
				// press R to reset animation
			case GLFW_KEY_R:
				startTime = std::chrono::high_resolution_clock::now();
				isPlaying = true;
				break;
				// escape to close window
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, true);
				break;
				// f1 to switch to user camera
			case GLFW_KEY_F1:
				isDebugCameraActive = false;
				isRenderCameraActive = false;
				isUserCameraActive = true;
				break;
				// f2 to switch to render camera (scene camera)
			case GLFW_KEY_F2:
				if (!cameras.empty()) {
					isDebugCameraActive = false;
					isRenderCameraActive = true;
					isUserCameraActive = false;
				}
				break;
				// F3 to switch to debug camera
			case GLFW_KEY_F3:
				isDebugCameraActive = true;
				break;
			}
		}
	}

	static void mouseCallback(GLFWwindow* window, double xposIn, double yposIn) {
		static float lastX = WIDTH / 2.0;
		static float lastY = HEIGHT / 2.0;
		static bool firstMouse = true;

		if (firstMouse) {
			lastX = xposIn;
			lastY = yposIn;
			firstMouse = false;
		}

		float xoffset = lastX - xposIn;
		float yoffset = lastY - yposIn;
		lastX = xposIn;
		lastY = yposIn;
		auto app = reinterpret_cast<FirstTriangle*>(glfwGetWindowUserPointer(window));
		if (app && !app->isDebugCameraActive) {
			app->userCamera.ProcessMouseMovement(xoffset, yoffset);
		}

	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<FirstTriangle*>(glfwGetWindowUserPointer(window));
		app->frameBufferResized = true;
	}

	void initVulkan() {
		createInstance();
		createSurface();

		pickPhysicalDevice();
		createLogicalDevice();

		createSwapChain();
		createImageViews();

		createRenderPass(swapChainImageFormat, swapChainRenderPass);
		createDescriptorSetLayout();
		auto bind = Vertex::getBindingDescription();
		auto attr = Vertex::getAttributeDescriptions();
		auto simpleBind = SimpleVertex::getBindingDescription();
		auto simpleAttr = SimpleVertex::getAttributeDescriptions();
		createPipelineLayout();
		createGraphicsPipeline(swapChainExtent, swapChainRenderPass, graphicsPipeline, bind, std::vector<VkVertexInputAttributeDescription>(attr.begin(), attr.end()), "shaders/vert.spv", "shaders/frag.spv");
		createGraphicsPipeline(swapChainExtent, swapChainRenderPass, simpleGraphicsPipeline, simpleBind, std::vector<VkVertexInputAttributeDescription>(simpleAttr.begin(), simpleAttr.end()), "shaders/simpleVert.spv", "shaders/simpleFrag.spv");

		createCommandPool();

		createSampler(textureSampler);
		createTextureImage(defaultImage, defaultImageMemory, "default.png");
		defaultImageView = createImageView(defaultImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
		loadAllTextures();

		createCubemap(cubemapImage, cubemapImageMemory, environment.radiance.src, cubemap);
		cubemapImageView = createImageView(cubemapImage, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 6, VK_IMAGE_VIEW_TYPE_CUBE);

		if (lambertian) {
			createLambertianCubeMap(inputFile, outputFile);
		}


		createCubemap(lambertianCubemapImage, lambertianCubemapImageMemory, "sky_lambertian.png", lambertianCubemap);
		lambertianCubemapImageView = createImageView(lambertianCubemapImage, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 6, VK_IMAGE_VIEW_TYPE_CUBE);


		createDepthResources(swapChainExtent);

		createFramebuffers();


		createVertexBuffersForAllMeshes(sceneGraph.meshes);
		// BUG
		//createIndexBuffer();



		createUniformBuffer();

		createDescriptorPool();
		createDescriptorSets();

		createCommandBuffers(swapChainCommandBuffers);
		createSyncObjects();
		
	}

	// skip window creation and swapchain
	void initVulkanHeadless() {
		createInstance();
		pickPhysicalDevice();
		createLogicalDevice();
		createImage(offscreenExtent.width, offscreenExtent.height, offscreenImageFormat, VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, offscreenImage, offscreenImageMemory);

		offscreenImageView = createImageView(offscreenImage, offscreenImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		createRenderPass(offscreenImageFormat, offscreenRenderPass);
		createDescriptorSetLayout();
		auto bind = Vertex::getBindingDescription();
		auto simpleBind = SimpleVertex::getBindingDescription();
		auto attr = Vertex::getAttributeDescriptions();
		auto simpleAttr = SimpleVertex::getAttributeDescriptions();
		createPipelineLayout();
		createGraphicsPipeline(swapChainExtent, swapChainRenderPass, graphicsPipeline, bind, std::vector<VkVertexInputAttributeDescription>(attr.begin(), attr.end()), "shaders/vert.spv", "shaders/frag.spv");
		createGraphicsPipeline(swapChainExtent, swapChainRenderPass, simpleGraphicsPipeline, simpleBind, std::vector<VkVertexInputAttributeDescription>(simpleAttr.begin(), simpleAttr.end()), "shaders/simpleVert.spv", "shaders/simpleFrag.spv");

		createCommandPool();
		createDepthResources(offscreenExtent);
		createOffscreenFrameBuffer();
		createVertexBuffersForAllMeshes(sceneGraph.meshes);
		createIndexBuffer();
		createUniformBuffer();
		createDescriptorPool();
		createDescriptorSets();
		createCommandBuffers(headlessCommandBuffers);
		createSyncObjects();
	}

	void loadAllTextures() {
		for (auto& material : materials) {
			// load normal maps
			loadTexture(material.normalMap, material.normalMap.src);
			loadTexture(material.displacementMap, material.displacementMap.src);

			// load textures
			if (std::holds_alternative<PBR>(material.materialType)) {
				PBR& pbr = std::get<PBR>(material.materialType);
				if (std::holds_alternative<Texture>(pbr.albedo)) {
					loadTexture(std::get<Texture>(pbr.albedo), std::get<Texture>(pbr.albedo).src);
				}
				/*if (std::holds_alternative<Texture>(pbr.roughness)) {
					loadTexture(std::get<Texture>(pbr.roughness), std::get<Texture>(pbr.roughness).src);
				}
				if (std::holds_alternative<Texture>(pbr.metalness)) {
					loadTexture(std::get<Texture>(pbr.metalness), std::get<Texture>(pbr.metalness).src);
				}*/
			}
			else if (std::holds_alternative<Lambertian>(material.materialType)) {
				Lambertian& lambertian = std::get<Lambertian>(material.materialType);
				if (std::holds_alternative<Texture>(lambertian.albedo)) {
					loadTexture(std::get<Texture>(lambertian.albedo), std::get<Texture>(lambertian.albedo).src);
				}
			}
		}
	}

	void loadTexture(Texture& texture, const std::string& fileName) {
		if (fileName.empty()) {
			return;
		}
		texture.src = fileName;

		// put texture creation process together
		createTextureImage(texture.image, texture.memory, fileName);
		texture.imageView = createImageView(texture.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
		texture.sampler = textureSampler;
	}


	void createLambertianCubeMap(const std::string& inFile, const std::string& outFile) {
		int width, height, channels;
		stbi_uc* pixel = stbi_load(inFile.c_str(), &width, &height, &channels, STBI_rgb_alpha);
		if (!pixel) {
			std::cerr << "Error loading input cubemap" << std::endl;
			return;
		}

		int totalPixels = width * height;
		std::vector<std::pair<float, int>> brightness_indices(totalPixels);
		glm::vec4* rgbData = new glm::vec4[totalPixels];

		// calculate brightness and store
		for (int i = 0; i < totalPixels; ++i) {
			glm::u8vec4 col(pixel[i * 4], pixel[i * 4 + 1], pixel[i * 4 + 2], pixel[i * 4 + 3]);
			rgbData[i] = rgbe_to_float(col);
			float brightness = 0.299f * rgbData[i].r + 0.587f * rgbData[i].g + 0.114f * rgbData[i].b;
			brightness_indices[i] = std::make_pair(brightness, i);
		}


		// descend sorting
		std::sort(brightness_indices.begin(), brightness_indices.end(), [](const auto& a, const auto& b) {
			return a.first > b.first;
			});

		// change the lightest 10000 pixels into dark grey
		const int maxBrightPixels = std::min(10000, totalPixels);
		for (int i = 0; i < maxBrightPixels; ++i) {
			int pixelIndex = brightness_indices[i].second;
			rgbData[pixelIndex].r = static_cast<unsigned char>(0.2f * 255); // R
			rgbData[pixelIndex].g = static_cast<unsigned char>(0.2f * 255); // G
			rgbData[pixelIndex].b = static_cast<unsigned char>(0.2f * 255); // B
			// Alpha unchange
		}


		cubemap.width = width;
		cubemap.height = height;
		// deep copy rgbdata to cubemap.data
		cubemap.data = new glm::vec4[width * height];
		std::memcpy(cubemap.data, rgbData, width * height * sizeof(glm::vec4));

		int outWidth = 16;
		int outHeight = 96;
		glm::vec4* outputData = new glm::vec4[outWidth * outHeight];

		int faceHeight = outHeight / 6;

		for (int y = 0; y < outHeight; ++y) {
			int faceIndex = y / faceHeight;
			float faceV = (y % faceHeight + 0.5f) / faceHeight;
			for (int x = 0; x < outWidth; ++x) {
				// convert uv to normalized device coordinates
				float u = (x + 0.5f) / outWidth;
				float v = faceV;

				glm::vec3 direction = faceIndexToDirection(faceIndex, u, v);

				//std::cout << direction.x << " " << direction.y << " " << direction.z << "\n";


				// initialize accmulated color
				glm::vec3 accumulatedColor(0.0f, 0.0f, 0.0f);

				int numSamples = 1000;
				for (int sample = 0; sample < numSamples; ++sample) {
					// calculate random direction on the hemisphere
					glm::vec3 sampleDirection = randomHemisphereSample(direction);

					// get color value
					glm::vec3 color = sampleCubemapDirection(cubemap, sampleDirection);

					// apply Lambertian's cosine law
					float cosineWeight = std::max(glm::dot(direction, sampleDirection), 0.0f);
					accumulatedColor += color * cosineWeight;
				}

				// normalize color
				accumulatedColor /= numSamples;

				int pixelIndex = y * outWidth + x;

				outputData[pixelIndex] = float_to_rgbe(accumulatedColor);

			}
		}

		std::vector<unsigned char> imageData(outWidth * outHeight * 4);

		for (int i = 0; i < outWidth * outHeight; ++i) {
			imageData[i * 4 + 0] = static_cast<unsigned char>(outputData[i].r); // R
			imageData[i * 4 + 1] = static_cast<unsigned char>(outputData[i].g); // G
			imageData[i * 4 + 2] = static_cast<unsigned char>(outputData[i].b); // B
			imageData[i * 4 + 3] = static_cast<unsigned char>(outputData[i].a); // e
			//std::cout << outputData[i].r << " " << outputData[i].g << " " << outputData[i].b << " " << outputData[i].a << "imageData\n";
		}


		// save image to file
		if (!stbi_write_png(outFile.c_str(), outWidth, outHeight, 4, imageData.data(), outWidth * 4)) {
			std::cerr << "Error saving output cubemap" << std::endl;
			return;
		}

		// clean image
		stbi_image_free(pixel);
		delete[] outputData;
	}

	void createCubemap(VkImage& cubemapImage, VkDeviceMemory& cubemapImageMemory, std::string imagePath, Cubemap& cubemap) {

		// load all six images and calculate total size
		int width, height, channels;
		std::string fullPath = "./model/" + imagePath;
		stbi_uc* pixel = stbi_load(fullPath.c_str(), &width, &height, &channels, STBI_rgb_alpha);

		if (!pixel) {
			std::cout << imagePath << "\n";
			throw std::runtime_error("failed to load cubemap image strip!");
		}

		if (width * 6 != height) {
			throw std::runtime_error("Image strip doesn't have the correct dimensions for a cubemap!");
		}


		VkDeviceSize imageSize = width * height * sizeof(glm::vec4);

		// decode rgbe and convert to rgb
		glm::vec4* rgbData = new glm::vec4[width * height];

		for (int i = 0; i < width * height; ++i) {
			glm::u8vec4 col(pixel[i * 4], pixel[i * 4 + 1], pixel[i * 4 + 2], pixel[i * 4 + 3]);
			rgbData[i] = rgbe_to_float(col);
		}



		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, rgbData, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		stbi_image_free(pixel);
		delete[] rgbData;

		createImage(width, height / 6, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, cubemapImage, cubemapImageMemory, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, 6);
		transitionImageLayout(cubemapImage, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6);

		copyBufferToImage(stagingBuffer, cubemapImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height / 6), 6);
		transitionImageLayout(cubemapImage, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 6);

		// clean up buffer & memory
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);

	}

	void createSampler(VkSampler& sampler) {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR; // VK_FILTER_NEAREST or VK_FILTER_LINEAR
		samplerInfo.minFilter = VK_FILTER_LINEAR;

		// VK_SAMPLER_ADDRESS_MODE_REPEAT
		// VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT
		// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		// VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE
		// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);

		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

		// white/black/transparent
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;

		// [0, texWidth) and [0, texHeight) or [0, 1) <- we want this
		samplerInfo.unnormalizedCoordinates = VK_FALSE;

		// used in shadow maps
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;

		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	void createTextureImage(VkImage& textureImage, VkDeviceMemory& textureImageMemory, std::string fileName) {
		int texWidth, texHeight, texChannels;
		std::string fullPath = "./model/" + fileName;
		stbi_uc* pixels = stbi_load(fullPath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
		VkDeviceSize imageSize = texWidth * texHeight * 4;

		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}

		// create a buffer in host visible memory
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		// copy the pixel values to the buffer
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		// clean up pixel array
		stbi_image_free(pixels);

		createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

		// Transition the texture image to VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
		// Execute the buffer to image copy operation
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		// transition to prepare it for shader access
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// clean up buffer & memory
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height, int layerCount = 1) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = layerCount;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(
			commandBuffer,
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);

		endSingleTimeCommands(commandBuffer);
	}

	void copyImageToBuffer(VkBuffer& dstBuffer, VkImage& image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();
		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dstBuffer, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, int layerCount = 1) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		// perform layout transitions
		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;

		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		// does not have mipmapping levels
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = layerCount;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}


		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);

	}

	void createDepthResources(VkExtent2D& extent) {
		VkFormat depthFormat = findDepthFormat();
		createImage(extent.width, extent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
	}

	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, VkImageCreateFlags flags = 0, uint32_t arrayLayers = 1) {
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = arrayLayers;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.flags = flags;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	// abstraction from createimageview class
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, int layerCount = 1, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D) {
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = viewType;
		viewInfo.format = format;
		// turn aspectMask to a parameter so we can change it for different image types

		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = layerCount;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image view!");
		}

		return imageView;
	}

	// takes a list of candidate formats in order from most desirable to least desirable
	// checks which is the first one that is supported
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

		// linearTilingFeatures: Use cases that are supported with linear tiling
		// optimalTilingFeatures: Use cases that are supported with optimal tiling
		// bufferFeatures : Use cases that are supported for buffers
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}
		throw std::runtime_error("failed to find supported format!");
	}
	// helper function, select a format with a depth component that supports usage as depth attachment
	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	// helper function, tells us if the chosen depth format contains a stencil component
	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void createDescriptorSets() {
		// create one descriptor set for each material in each frame in flight, all with the same layout

		size_t totalMaterials = materials.size();
		std::cout << "total materials" << totalMaterials << "\n";
		size_t totalBuffers = totalMaterials * MAX_FRAME_IN_FLIGHT;

		std::vector<VkDescriptorSetLayout> layouts(totalBuffers, descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(totalBuffers);
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(totalBuffers);
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t frame = 0; frame < MAX_FRAME_IN_FLIGHT; frame++) {
			for (size_t matIndex = 0; matIndex < materials.size(); matIndex++) {
				Material& material = materials[matIndex];
				size_t descriptorIndex = frame * materials.size() + matIndex;
				VkDescriptorSet descriptorSet = descriptorSets[descriptorIndex];

				VkDescriptorBufferInfo bufferInfo{};
				bufferInfo.buffer = uniformBuffers[frame];
				bufferInfo.offset = 0;
				bufferInfo.range = sizeof(UniformBufferObject);

				// give texture & normal map a default value if not using
				VkDescriptorImageInfo defaultImageInfo{};
				defaultImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				defaultImageInfo.imageView = defaultImageView;
				defaultImageInfo.sampler = textureSampler;

				VkDescriptorImageInfo cubemapImageInfo{};
				cubemapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				cubemapImageInfo.imageView = cubemapImageView;
				cubemapImageInfo.sampler = textureSampler; // i use the same sampler for cubemap

				VkDescriptorImageInfo lambertianCubemapImageInfo{};
				lambertianCubemapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				lambertianCubemapImageInfo.imageView = lambertianCubemapImageView;
				lambertianCubemapImageInfo.sampler = textureSampler;

				std::array<VkWriteDescriptorSet, 5> descriptorWrites{};

				descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[0].dstSet = descriptorSet;
				descriptorWrites[0].dstBinding = 0;
				descriptorWrites[0].dstArrayElement = 0;
				descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				descriptorWrites[0].descriptorCount = 1;
				descriptorWrites[0].pBufferInfo = &bufferInfo;

				// initialize these two with default image view
				descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[1].dstSet = descriptorSet;
				descriptorWrites[1].dstBinding = 1;
				descriptorWrites[1].dstArrayElement = 0;
				descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[1].descriptorCount = 1;
				descriptorWrites[1].pImageInfo = &defaultImageInfo;

				descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[2].dstSet = descriptorSet;
				descriptorWrites[2].dstBinding = 2;
				descriptorWrites[2].dstArrayElement = 0;
				descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[2].descriptorCount = 1;
				descriptorWrites[2].pImageInfo = &defaultImageInfo;

				if (std::holds_alternative<Lambertian>(material.materialType)) {
					Lambertian& lambertian = std::get<Lambertian>(material.materialType);
					if (std::holds_alternative<Texture>(lambertian.albedo)) {
						Texture& tex = std::get<Texture>(lambertian.albedo);
						VkDescriptorImageInfo imageInfo{};
						imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
						imageInfo.imageView = tex.imageView;
						imageInfo.sampler = tex.sampler;

						descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						descriptorWrites[1].dstSet = descriptorSet;
						descriptorWrites[1].dstBinding = 1;
						descriptorWrites[1].dstArrayElement = 0;
						descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
						descriptorWrites[1].descriptorCount = 1;
						descriptorWrites[1].pImageInfo = &imageInfo;
					}
				}
				else if (std::holds_alternative<PBR>(material.materialType)) {
					PBR& pbr = std::get<PBR>(material.materialType);
					if (std::holds_alternative<Texture>(pbr.albedo)) {
						Texture& tex = std::get<Texture>(pbr.albedo);
						VkDescriptorImageInfo imageInfo{};
						imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
						imageInfo.imageView = tex.imageView;
						imageInfo.sampler = tex.sampler;

						descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
						descriptorWrites[1].dstSet = descriptorSet;
						descriptorWrites[1].dstBinding = 1;
						descriptorWrites[1].dstArrayElement = 0;
						descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
						descriptorWrites[1].descriptorCount = 1;
						descriptorWrites[1].pImageInfo = &imageInfo;
					}
				}

				if (!material.normalMap.src.empty()) {
					VkDescriptorImageInfo normalmapImageInfo{};
					normalmapImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
					normalmapImageInfo.imageView = material.normalMap.imageView;
					normalmapImageInfo.sampler = material.normalMap.sampler;

					descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					descriptorWrites[2].dstSet = descriptorSet;
					descriptorWrites[2].dstBinding = 2;
					descriptorWrites[2].dstArrayElement = 0;
					descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
					descriptorWrites[2].descriptorCount = 1;
					descriptorWrites[2].pImageInfo = &normalmapImageInfo;
				}

				descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[3].dstSet = descriptorSets[descriptorIndex];
				descriptorWrites[3].dstBinding = 3;
				descriptorWrites[3].dstArrayElement = 0;
				descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[3].descriptorCount = 1;
				descriptorWrites[3].pImageInfo = &cubemapImageInfo;

				descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
				descriptorWrites[4].dstSet = descriptorSets[descriptorIndex];
				descriptorWrites[4].dstBinding = 4;
				descriptorWrites[4].dstArrayElement = 0;
				descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				descriptorWrites[4].descriptorCount = 1;
				descriptorWrites[4].pImageInfo = &lambertianCubemapImageInfo;

				vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
			}
		}
	}

	void createDescriptorPool() {
		//  allocate one of these descriptors for every frame
		size_t totalMaterials = materials.size();
		std::array<VkDescriptorPoolSize, 5> poolSizes{};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAME_IN_FLIGHT * totalMaterials);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAME_IN_FLIGHT * totalMaterials);
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAME_IN_FLIGHT * totalMaterials);
		poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[3].descriptorCount = static_cast<uint32_t>(MAX_FRAME_IN_FLIGHT * totalMaterials);
		poolSizes[4].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[4].descriptorCount = static_cast<uint32_t>(MAX_FRAME_IN_FLIGHT * totalMaterials);


		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAME_IN_FLIGHT * totalMaterials * 5);

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	void createUniformBuffer() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		size_t totalBuffers = MAX_FRAME_IN_FLIGHT;
		uniformBuffers.resize(totalBuffers);
		uniformBuffersMemory.resize(totalBuffers);
		uniformBuffersMapped.resize(totalBuffers);

		// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
		// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		for (size_t i = 0; i < MAX_FRAME_IN_FLIGHT; i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

			// map the buffer right after creation to get a pointer to which we can write the data later on
			// "persistent mapping"
			vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}

	};

	void createDescriptorSetLayout() {
		// every binding needs to be described through a VkDescriptorSetLayoutBinding struct
		// ubo
		VkDescriptorSetLayoutBinding uboLayoutBinding{};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		// could be an array of ubo in terms of each bones in skeletal animation, i.e.
		uboLayoutBinding.descriptorCount = 1;
		// which shader stages the descriptor is going to be referenced
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		// albedo
		VkDescriptorSetLayoutBinding samplerLayoutBinding{};
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// normalmap
		VkDescriptorSetLayoutBinding normalmapLayoutBinding{};
		normalmapLayoutBinding.binding = 2;
		normalmapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		normalmapLayoutBinding.descriptorCount = 1;
		normalmapLayoutBinding.pImmutableSamplers = nullptr;
		normalmapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// cubemap
		VkDescriptorSetLayoutBinding cubemapLayoutBinding{};
		cubemapLayoutBinding.binding = 3;
		cubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		cubemapLayoutBinding.descriptorCount = 1;
		cubemapLayoutBinding.pImmutableSamplers = nullptr;
		cubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		// lambertian cubemap
		VkDescriptorSetLayoutBinding lambertianCubemapLayoutBinding{};
		lambertianCubemapLayoutBinding.binding = 4;
		lambertianCubemapLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		lambertianCubemapLayoutBinding.descriptorCount = 1;
		lambertianCubemapLayoutBinding.pImmutableSamplers = nullptr;
		lambertianCubemapLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		std::array<VkDescriptorSetLayoutBinding, 5> bindings = { uboLayoutBinding, samplerLayoutBinding, normalmapLayoutBinding, cubemapLayoutBinding, lambertianCubemapLayoutBinding };
		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
		// VK_BUFFER_USAGE_TRANSFER_SRC_BIT: Buffer can be used as source in a memory transfer operation.
		// VK_BUFFER_USAGE_TRANSFER_DST_BIT: Buffer can be used as destination in a memory transfer operation.

		// copy data from the stagingBuffer to the vertexBuffer
		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		//clean up
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	void createVertexBuffer(Mesh& mesh) {
		Attribute attr = mesh.getAttribute("POSITION");
		int stride = attr.stride;
		bool isSimple;

		if (stride == 28) {
			isSimple = true;
		}
		else {
			isSimple = false;
		}

		std::vector<Vertex> vertices;
		std::vector<SimpleVertex> simpleVertices;
		VkDeviceSize bufferSize;

		if (!isSimple) {
			vertices = parseVertices(attr.src, mesh);
			//printVertices(vertices);
			bufferSize = sizeof(vertices[0]) * vertices.size();
		}
		else {
			simpleVertices = parseSimpleVertices(attr.src, mesh);
			bufferSize = sizeof(simpleVertices[0]) * simpleVertices.size();
		};

		// vertex buffer only uses a host visible buffer as temporary buffer
		// uses a device local one as actual vertex buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		if (!isSimple) {
			memcpy(data, vertices.data(), (size_t)bufferSize);
		}
		else {
			memcpy(data, simpleVertices.data(), (size_t)bufferSize);
		}

		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mesh.vertexBuffer, mesh.vertexBufferMemory);
		// VK_BUFFER_USAGE_TRANSFER_SRC_BIT: Buffer can be used as source in a memory transfer operation.
		// VK_BUFFER_USAGE_TRANSFER_DST_BIT: Buffer can be used as destination in a memory transfer operation.

		// copy data from the stagingBuffer to the vertexBuffer
		copyBuffer(stagingBuffer, mesh.vertexBuffer, bufferSize);

		//clean up
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void createVertexBuffersForAllMeshes(std::vector<Mesh>& meshes) {
		for (Mesh& mesh : meshes) {
			createVertexBuffer(mesh);
		}
	}

	void printVertices(const std::vector<Vertex>& vertices) {
		for (const auto& vertex : vertices) {
			std::cout << "Vertex Position: "
				<< "X: " << vertex.texCoord.x << ", "
				<< "Y: " << vertex.texCoord.y << ", ";
			//<< "Z: " << vertex.pos.z << std::endl;
		}
	}

	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memoryProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
			if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0) {
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}
		vkDeviceWaitIdle(device);

		cleanupSwapChain();

		// make sure every frame/image related thing is recreated
		createSwapChain();
		createImageViews();
		createDepthResources(swapChainExtent);
		createFramebuffers();
	}

	void cleanupSwapChain() {
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		for (auto framebuffer : swapChainFramebuffers) {
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		for (auto imageView : swapChainImageViews) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAME_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAME_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAME_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (int i = 0; i < MAX_FRAME_IN_FLIGHT; ++i) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create semaphores/fences!");
			}
		}
	}

	void createCommandBuffers(std::vector<VkCommandBuffer>& commandBuffers) {
		commandBuffers.resize(MAX_FRAME_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffers!");
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex, VkRenderPass renderPass) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}
		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		// The range of depths in the depth buffer is 0.0 to 1.0 in Vulkan
		// 1.0 lies at the far view plane and 0.0 at the near view plane
		// The initial value at each point in the depth buffer should be the furthest possible depth, which is 1.0.
		clearValues[1].depthStencil = { 1.0f, 0 };

		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);



		//  can only have a single index buffer
		//vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swapChainExtent.width);
		viewport.height = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		// render frame
		for (int rootGlobalIndex : sceneGraph.parsedScene.roots) {
			int containerIndex = sceneGraph.globalNodeIndexMap.at(rootGlobalIndex); // directly get rootGlobalIndex
			const Node& rootNode = sceneGraph.nodes.at(containerIndex);
			glm::mat4 parentMatrix = glm::mat4(1.0f);
			renderNode(rootNode, rootGlobalIndex, commandBuffer, currentFrame, parentMatrix);
		}

		//vkCmdDrawIndexed(commandBuffer, indices.size(), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void recordCommandBufferHeadless(VkCommandBuffer& commandBuffer) {
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin recording command buffer!");
		}
		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = offscreenRenderPass; // Use offscreen render pass
		renderPassInfo.framebuffer = offscreenFramebuffer; // Use the offscreen framebuffer
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = offscreenExtent;

		std::array<VkClearValue, 2> clearValues{};
		clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
		// The range of depths in the depth buffer is 0.0 to 1.0 in Vulkan
		// 1.0 lies at the far view plane and 0.0 at the near view plane
		// The initial value at each point in the depth buffer should be the furthest possible depth, which is 1.0.
		clearValues[1].depthStencil = { 1.0f, 0 };

		renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
		renderPassInfo.pClearValues = clearValues.data();
		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		//vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		//  can only have a single index buffer
		//vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(offscreenExtent.width);
		viewport.height = static_cast<float>(offscreenExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = offscreenExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		// specify if we want to bind descriptor sets to the graphics or compute pipeline
		//size_t totalBuffers = sceneGraph.meshes.size() * MAX_FRAME_IN_FLIGHT;

		// render frame
		for (int rootGlobalIndex : sceneGraph.parsedScene.roots) {
			int containerIndex = sceneGraph.globalNodeIndexMap.at(rootGlobalIndex); // directly get rootGlobalIndex
			const Node& rootNode = sceneGraph.nodes.at(containerIndex);
			glm::mat4 parentMatrix = glm::mat4(1.0f);
			renderNode(rootNode, rootGlobalIndex, commandBuffer, currentFrame, parentMatrix);
		}


		//vkCmdDrawIndexed(commandBuffer, indices.size(), 1, 0, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer!");
		}
	}

	void renderNode(const Node& node, const int& globalNodeIndex, VkCommandBuffer commandBuffer, int currentFrame, glm::mat4& parentMatrix) {

		glm::mat4 modelMatrix;
		int nodeIndex = sceneGraph.getNodeIndex(sceneGraph.nodes, node);

		if (nodeHasAnimation(globalNodeIndex)) {
			// get clip
			std::vector<AnimationClip> matchedClips = getAnimationClipsForNode(globalNodeIndex);

			// model matrix = scale * rotation * translation
			if (!matchedClips.empty()) {
				modelMatrix = calculateModelMatrix(node);

				glm::mat4 animatedMatrix = calculateModelMatrixForNodeAtTime(matchedClips, animTime, modelMatrix, nodes[nodeIndex]);
				modelMatrix = animatedMatrix;
			}
		}
		else {
			modelMatrix = calculateModelMatrix(node);
		}

		// apply parent nodes
		modelMatrix = parentMatrix * modelMatrix;

		glm::mat4 viewMat;
		glm::mat4 projMat;
		glm::mat4 vpMat;
		// getting vp matrix

		if (isUserCameraActive) {
			viewMat = userCamera.GetViewMatrix();
			projMat = userCamera.GetProjectionMatrix();

		}
		else if (isRenderCameraActive) {
			viewMat = renderCamera.GetViewMatrix();
			projMat = renderCamera.GetProjectionMatrix();
		}
		vpMat = projMat * viewMat;

		// frustum culling
		if (node.mesh >= 0) {
			BoundingBox box = box.calculateBoundingBox(node, sceneGraph, modelMatrix);
			if (!isInsideFrustum(box, vpMat)) {
				return;
			}
		}

		// calculate materialIndex
		int materialIndex = 0;
		Material material;
		int materialType = 0;
		if (node.mesh >= 0) {
			int meshContainerIndex = sceneGraph.meshIndexMap.at(node.mesh);
			const Mesh& mesh = sceneGraph.meshes.at(meshContainerIndex);

			Attribute attr = mesh.getAttribute("POSITION");
			int stride = attr.stride;

			if (stride == 52) {
			materialIndex = sceneGraph.materialIndexMap.at(mesh.material);
			material = sceneGraph.materials.at(materialIndex);
			materialType = material.getMaterialType();
			}		
		}

		// calculate descriptor set index 
		int descriptorSetIndex = currentFrame * materials.size() + materialIndex;

		// bind descriptor
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[descriptorSetIndex], 0, nullptr);

		// update ubo & push constants
		pushConstants.albedoColor = glm::vec4(-1);
		pushConstants.roughness = 0;
		pushConstants.metalness = 0;

		if (materialType == 1) {
			Lambertian& lambertian = std::get<Lambertian>(material.materialType);
			if (std::holds_alternative<glm::vec3>(lambertian.albedo)) {
				pushConstants.albedoColor = glm::vec4(std::get<glm::vec3>(lambertian.albedo), 1);
			}
		}
		else if (materialType == 4) {
			PBR& pbr = std::get<PBR>(material.materialType);
			if (std::holds_alternative<glm::vec3>(pbr.albedo)) {
				pushConstants.albedoColor = glm::vec4(std::get<glm::vec3>(pbr.albedo), 1);
			}
			if (std::holds_alternative<float>(pbr.roughness)) {
				pushConstants.roughness = std::get<float>(pbr.roughness);
			}
			if (std::holds_alternative<float>(pbr.metalness)) {
				pushConstants.metalness = std::get<float>(pbr.metalness);
			}
		}
		pushConstants.model = modelMatrix;
		pushConstants.materialType = materialType;
		vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pushConstants);
		updateUniformBuffer(currentFrame, nodeIndex);

		if (node.mesh >= 0) {

			try {
				if (materialType == 0) { vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, simpleGraphicsPipeline); }
				else {
					vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
				};

				int containerIndex = sceneGraph.meshIndexMap.at(node.mesh);
				const Mesh& mesh = sceneGraph.meshes.at(containerIndex);

				VkBuffer buffers[] = { mesh.vertexBuffer };
				VkDeviceSize offsets[] = { 0 };
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, offsets);

				vkCmdDraw(commandBuffer, static_cast<uint32_t>(mesh.count), 1, 0, 0);

			}
			catch (const std::out_of_range& e) {
				std::cerr << "Out of Range error: " << e.what() << '\n';
			}
		}

		// add values to renderCamera
		if (node.camera >= 0) {
			renderCameraPosition = node.translation;
			renderCameraPitch = quatToPitch(node.rotation.w, node.rotation.x, node.rotation.y, node.rotation.z);
			renderCameraYaw = quatToYaw(node.rotation.x, node.rotation.y, node.rotation.z, node.rotation.w) + 90.0f;
		}

		// recursively render children nodes
		if (!node.children.empty()) {
			for (int childIndex : node.children) {
				int containerIndex = sceneGraph.globalNodeIndexMap.at(childIndex);
				const Node& childNode = sceneGraph.nodes.at(containerIndex); //
				renderNode(childNode, childIndex, commandBuffer, currentFrame, modelMatrix);
			}
		}

	}

	float quatToPitch(float w, float x, float y, float z) {
		float pitch = std::asin(2.0f * (w * y - z * x));
		const float radianToDegree = 180.0f / M_PI;
		pitch *= radianToDegree;
		return pitch;
	}

	float quatToYaw(float w, float x, float y, float z) {
		float  yaw = std::atan2(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y));
		const float radianToDegree = 180.0f / M_PI;
		yaw *= radianToDegree;
		return yaw;
	}

	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);


		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());
		for (size_t i = 0; i < swapChainImageViews.size(); i++) {
			std::array<VkImageView, 2> attachments = {
			swapChainImageViews[i],
			depthImageView
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = swapChainRenderPass;
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
			framebufferInfo.pAttachments = attachments.data();
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}

	void createOffscreenFrameBuffer() {
		VkImageView attachments[] = {
	   offscreenImageView,
	   depthImageView
		};
		VkFramebufferCreateInfo framebufferInfo{};
		framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebufferInfo.renderPass = offscreenRenderPass;
		framebufferInfo.attachmentCount = 2;
		framebufferInfo.pAttachments = attachments;
		framebufferInfo.width = offscreenExtent.width;
		framebufferInfo.height = offscreenExtent.height;
		framebufferInfo.layers = 1;

		if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &offscreenFramebuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create offscreen framebuffer!");
		}
	}

	void createRenderPass(VkFormat& colorFormat, VkRenderPass& renderPass) {
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = colorFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		if (headless) {
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		}
		else {
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		}

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		// include a depth attachment
		VkAttachmentDescription depthAttachment{};
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		VkAttachmentReference depthAttachmentRef{};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;
		subpass.pDepthStencilAttachment = &depthAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;

		// refer to both attachments
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
		renderPassInfo.pAttachments = attachments.data();
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create render pass!");
		}
	}

	void createGraphicsPipeline(VkExtent2D& extent, VkRenderPass& renderPass, VkPipeline& graphicsPipeline,
		VkVertexInputBindingDescription& vertexBindingDescriptions,
		const std::vector<VkVertexInputAttributeDescription>& vertexAttributeDescriptions,
		std::string vertShader, std::string fragShader)
	{

		auto vertShaderCode = readFile(vertShader);
		auto fragShaderCode = readFile(fragShader);

		auto bindingDescription = vertexBindingDescriptions;
		auto attributeDescriptions = vertexAttributeDescriptions;

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";
		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";
		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)extent.width;
		viewport.height = (float)extent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = extent;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT; // backside will be culled
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; // y is flipped so vertices are drawn counter clockwisely
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkPipelineDepthStencilStateCreateInfo depthStencil{};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		// specifies if the depth of new fragments should be compared to the depth buffer
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		// specifies the comparison that is performed to keep or discard fragments, lower depth = closer
		depthStencil.depthBoundsTestEnable = VK_FALSE;

		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;
		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDynamicState = &dynamicState;
		pipelineInfo.layout = pipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;
		pipelineInfo.pDepthStencilState = &depthStencil;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createPipelineLayout() {
		// specify the descriptor set layout during pipeline creation, reference the layout object
		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;;
		pushConstantRange.offset = 0;
		pushConstantRange.size = 104;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create pipeline layout!");
		}
	}

	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];
			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;
			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;
			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create image views!");
			}
		}
	}

	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
		if (!headless) {
			uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

			if (indices.graphicsFamily != indices.presentFamily) {
				createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices = queueFamilyIndices;
			}
			else {
				createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
				createInfo.queueFamilyIndexCount = 0;
				createInfo.pQueueFamilyIndices = nullptr;
			}
		}


		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	void createLogicalDevice() {
		// request anisotropic function
		VkPhysicalDeviceFeatures deviceFeatures{};
		deviceFeatures.samplerAnisotropy = VK_TRUE;

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies;
		if (!headless) { uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() }; }
		else {
			uniqueQueueFamilies = { indices.graphicsFamily.value() };
		}


		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		queueCreateInfo.queueCount = 1;

		queueCreateInfo.pQueuePriorities = &queuePriority;

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();

		createInfo.pEnabledFeatures = &deviceFeatures;

		if (!headless) {
			createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
			createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		}


		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		if (!headless) {
			vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
		}

	}

	void pickPhysicalDevice() {

		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			if (!headless) {
				VkBool32 presentSupport = false;
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
				if (presentSupport) {
					indices.presentFamily = i;
				}
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}

		return indices;
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}
		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}
		return shaderModule;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device);
		bool extensionSupported = checkDeviceExtensionSupport(device);
		if (!headless) {
			bool swapChainAdequate = false;
			if (extensionSupported) {
				SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
				swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
			}
			VkPhysicalDeviceFeatures supportedFeatures;
			vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
			return extensionSupported && indices.isComplete() && swapChainAdequate && supportedFeatures.samplerAnisotropy;
		}
		else {
			return indices.isComplete();
		}

	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {

		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());


		for (const char* extensionName : deviceExtensions) {
			bool extensionFound = false;

			for (const auto& extensionProperties : availableExtensions) {
				if (strcmp(extensionName, extensionProperties.extensionName) == 0) {
					extensionFound = true;
					break;
				}
			}

			if (!extensionFound) {
				return false;
			}
		}

		return true;
	}

	// "update"
	void mainLoop() {
		startTime = std::chrono::high_resolution_clock::now();

		while (!glfwWindowShouldClose(window)) {
			auto currentTime = std::chrono::high_resolution_clock::now();
			if (isPlaying) {
				animTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
				//std::cout << animTime << '\n';
			}

			deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
			lastTime = currentTime;
			userCamera.ProcessKeyboard(window, deltaTime);

			debugCamera.setCameraTransform(debugCameraPosition, debugCameraYaw, debugCameraPitch);

			renderCamera.setCameraTransform(renderCameraPosition, renderCameraYaw, renderCameraPitch);

			glfwPollEvents();
			drawFrame();
		}

		vkDeviceWaitIdle(device);
	}

	// handle events ("main loop" in headless mode)
	void handleEvents(const std::vector<Event>& events) {
		for (const auto& event : events) {
			switch (event.type) {
			case EventType::AVAILABLE:
				if (isPlaying) {
					// using time stamp in AVAILABLE events to update animTime
					long long elapsedTime = event.timestamp - animationStartTime;
					float animationTime = elapsedTime * playbackRate / 1e6f; // convert to seconds
					animTime = animationTime;
				}
				//std::cout << animTime << '\n';
				makeImageAvailable();
				break;
			case EventType::PLAY:
				if (event.params.size() >= 2) {
					animationStartTime = std::stof(event.params[0]);
					playbackRate = std::stof(event.params[1]);
					isPlaying = true;
				}
				break;
			case EventType::SAVE:
				if (!event.params.empty()) {
					const std::string& filename = event.params[0];
					saveCurrentImage(offscreenImage, offscreenImageFormat, offscreenExtent, filename);
				}
				break;
			case EventType::MARK:
				if (!event.params.empty()) {
					std::cout << "MARK ";
					for (const auto& word : event.params) {
						std::cout << word << " ";
					}
					std::cout << std::endl;
				}
				break;
			}
		}
	}

	void makeImageAvailable() {
		renderCamera.setCameraTransform(renderCameraPosition, renderCameraYaw, renderCameraPitch);
		drawFrameHeadless();
		vkDeviceWaitIdle(device);
	}

	void saveCurrentImage(VkImage& offscreenImage, VkFormat& imageFormat, VkExtent2D& extent, const std::string& filename) {
		VkDeviceSize imageSize = static_cast<VkDeviceSize>(extent.width) * extent.height * 4; // RGBA
		VkDeviceSize rgbSize = static_cast<VkDeviceSize>(extent.width) * extent.height * 3; // RGB
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;

		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		transitionImageLayout(offscreenImage, offscreenImageFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
		copyImageToBuffer(stagingBuffer, offscreenImage, offscreenExtent.width, offscreenExtent.height);

		// map values from staging buffer to CPU accessible area
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);

		unsigned char* pixelData = static_cast<unsigned char*>(data);

		// save data to PPM
		std::ofstream file(filename, std::ios::out | std::ios::binary);
		file << "P6\n" << extent.width << " " << extent.height << "\n255\n";
		for (uint32_t i = 0; i < imageSize; i += 4) {
			// ignore Alpha channel
			file.write(reinterpret_cast<const char*>(&pixelData[i]), 3);
		}

		file.close();

		// clean up
		vkUnmapMemory(device, stagingBufferMemory);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	void ProcessInput(GLFWwindow* window, float deltaTime) {
		userCamera.ProcessKeyboard(window, deltaTime);
	}

	// render frame
	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		uint32_t imageIndex;

		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
		if (result == VK_ERROR_OUT_OF_DATE_KHR) {
			frameBufferResized = false;
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
			throw std::runtime_error("failed to acquire swap chain image!");
		}


		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(swapChainCommandBuffers[currentFrame], 0);
		recordCommandBuffer(swapChainCommandBuffers[currentFrame], imageIndex, swapChainRenderPass);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &swapChainCommandBuffers[currentFrame];

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = signalSemaphores;

		VkSwapchainKHR swapChains[] = { swapChain };
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = swapChains;
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {
			throw std::runtime_error("failed to present swap chain image!");
		}

		currentFrame = (currentFrame + 1) & MAX_FRAME_IN_FLIGHT;
	}

	// draw frame in headless mode
	void drawFrameHeadless() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(headlessCommandBuffers[currentFrame], 0);
		recordCommandBufferHeadless(headlessCommandBuffers[currentFrame]);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &headlessCommandBuffers[currentFrame];


		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer");
		}


		currentFrame = (currentFrame + 1) & MAX_FRAME_IN_FLIGHT;
	}

	// debug matrix
	void printMatrix(const glm::mat4& matrix) {
		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				std::cout << matrix[i][j] << " ";
			}
			std::cout << "\n";
		}
		std::cout << "\n" << std::endl;
	}

	// if a node has animations
	bool nodeHasAnimation(int globalNodeIndex) {
		// find if nodeIndex in nodeAnimations map
		auto it = sceneGraph.nodeAnimations.find(globalNodeIndex);
		if (it != sceneGraph.nodeAnimations.end()) {
			// if so, find if it has clips
			return !it->second.clips.empty();
		}
		return false;
	}

	// find matched clips for globalNodeIndex in clips[]
	std::vector<AnimationClip> getAnimationClipsForNode(int globalNodeIndex) {
		std::vector<AnimationClip> matchedClips;
		for (const auto& clip : clips) {
			if (clip.node == globalNodeIndex) {
				matchedClips.push_back(clip);
			}
		}
		return matchedClips;
	}

	// calculate model matrix according to animations, and send it to GPU
	glm::mat4 calculateModelMatrixForNodeAtTime(const std::vector<AnimationClip>& clips, const float currentTime, const glm::mat4& inModelMat, const Node& node) {
		glm::mat4 modelMatrix = inModelMat;

		std::vector<AnimationClip> transClips;
		std::vector<AnimationClip> rotationClips;
		std::vector<AnimationClip> scaleClips;

		for (AnimationClip clip : clips) {
			if (clip.channel == "translation") { transClips.push_back(clip); }
			if (clip.channel == "rotation") { rotationClips.push_back(clip); }
			if (clip.channel == "scale") { scaleClips.push_back(clip); }
		}

		// find the nearest frames with current time
		// default is the last frame
		AnimationClip clip = clips[0];
		size_t index = 0;
		if (currentTime < clip.times.back()) {
			for (size_t i = 0; i < clip.times.size() - 1; ++i) {
				if (currentTime < clip.times[i + 1]) {
					index = i;
					break;
				}
			}
		}
		else {
			index = clip.times.size() - 1;
		}

		// calculate interpolation factor
		double t = 0;
		if (index < clip.times.size() - 1) {
			double duration = clip.times[index + 1] - clip.times[index];
			if (duration > 0) {
				t = (currentTime - clip.times[index]) / duration;
			}
		}
		else {
			// when currentTime > time in last frame, just keep last frame's transform
			t = 1;
			//return modelMatrix;
		}

		// calculate new model matrix according to the channel
		glm::mat4 translationAnim = glm::translate(glm::mat4(1.0f), node.translation);
		glm::mat4 rotationAnim = glm::toMat4(node.rotation);
		glm::mat4 scaleAnim = glm::scale(glm::mat4(1.0f), node.scale);

		for (AnimationClip clip : transClips) {
			if (index < clip.times.size() - 1) {

				glm::vec3 startValue{ clip.values[index * 3], clip.values[index * 3 + 1], clip.values[index * 3 + 2] };
				glm::vec3 endValue{ clip.values[(index + 1) * 3], clip.values[(index + 1) * 3 + 1], clip.values[(index + 1) * 3 + 2] };
				glm::vec3 interpolatedValue = lerp(startValue, endValue, t);
				translationAnim = glm::translate(glm::mat4(1.0f), interpolatedValue);

			}
			else {
				glm::vec3 interpolatedValue = glm::vec3(clip.values[(index) * 3], clip.values[(index) * 3 + 1], clip.values[(index) * 3 + 2]);
				translationAnim = glm::translate(glm::mat4(1.0f), interpolatedValue);
			}
		}

		for (AnimationClip clip : rotationClips) {
			if (index < clip.times.size() - 1) {

				glm::quat startValue{ static_cast<float>(clip.values[index * 4 + 3]), static_cast<float>(clip.values[index * 4]), static_cast<float>(clip.values[index * 4 + 1]), static_cast<float>(clip.values[index * 4 + 2]) };
				glm::quat endValue{ static_cast<float>(clip.values[(index + 1) * 4 + 3]), static_cast<float>(clip.values[(index + 1) * 4]), static_cast<float>(clip.values[(index + 1) * 4 + 1]), static_cast<float>(clip.values[(index + 1) * 4 + 2]) };
				glm::quat interpolatedValue = slerp(startValue, endValue, t);
				rotationAnim = glm::toMat4(interpolatedValue);

			}
			else {
				glm::quat interpolatedValue = glm::quat(static_cast<float>(clip.values[(index) * 4 + 3]), static_cast<float>(clip.values[(index) * 4]), static_cast<float>(clip.values[(index) * 4 + 1]), static_cast<float>(clip.values[(index) * 4 + 2]));
				rotationAnim = glm::toMat4(interpolatedValue);
			}
		}

		for (AnimationClip clip : scaleClips) {
			if (index < clip.times.size() - 1) {

				glm::vec3 startValue{ clip.values[index * 3], clip.values[index * 3 + 1], clip.values[index * 3 + 2] };
				glm::vec3 endValue{ clip.values[(index + 1) * 3], clip.values[(index + 1) * 3 + 1], clip.values[(index + 1) * 3 + 2] };
				glm::vec3 interpolatedValue = lerp(startValue, endValue, t);
				scaleAnim = glm::scale(glm::mat4(1.0f), interpolatedValue);

			}
			else {
				glm::vec3 interpolatedValue = glm::vec3(clip.values[(index) * 3], clip.values[(index) * 3 + 1], clip.values[(index) * 3 + 2]);
				scaleAnim = glm::scale(glm::mat4(1.0f), interpolatedValue);
			}
		}


		return translationAnim * rotationAnim * scaleAnim;
	}

	// helper functions: lerp & slerp
	glm::vec3 lerp(const glm::vec3& start, const glm::vec3& end, float t) {
		return start + t * (end - start);
	}

	glm::quat slerp(const glm::quat& start, const glm::quat& end, float t) {
		return glm::slerp(start, end, t);
	}

	void updateUniformBuffer(uint32_t currentImage, const int nodeIndex) {

		UniformBufferObject ubo{};

		if (isDebugCameraActive) {
			ubo.view = debugCamera.GetViewMatrix();
			ubo.projection = debugCamera.GetProjectionMatrix();
			ubo.cameraPos = glm::vec4(debugCamera.Position,0);

		}
		else if (isUserCameraActive) {
			ubo.view = userCamera.GetViewMatrix();
			ubo.projection = userCamera.GetProjectionMatrix();
			ubo.cameraPos = glm::vec4(userCamera.Position,0);
		}
		else if (isRenderCameraActive) {
			Camera sceneCamera = cameras[0];
			renderCamera.NearPlane = sceneCamera.perspective.near;
			renderCamera.FarPlane = sceneCamera.perspective.far;
			renderCamera.FOV = sceneCamera.perspective.vfov * 180.0f / M_PI;
			renderCamera.AspectRatio = sceneCamera.perspective.aspect;
			ubo.view = renderCamera.GetViewMatrix();
			ubo.projection = renderCamera.GetProjectionMatrix();
			ubo.cameraPos = glm::vec4(renderCamera.Position,0);

		}

		ubo.projection[1][1] *= -1;

		


		// GLM was originally designed for OpenGL, where the Y coordinate of the clip coordinates is inverted;

		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo)); //we only map the uniform buffer once

	}

	void cleanupTexture(Texture& texture) {
		if (texture.imageView != VK_NULL_HANDLE) {
			vkDestroyImageView(device, texture.imageView, nullptr);
			texture.imageView = VK_NULL_HANDLE;
		}
		if (texture.image != VK_NULL_HANDLE) {
			vkDestroyImage(device, texture.image, nullptr);
			texture.image = VK_NULL_HANDLE;
		}
		if (texture.memory != VK_NULL_HANDLE) {
			vkFreeMemory(device, texture.memory, nullptr);
			texture.memory = VK_NULL_HANDLE;
		}
	}

	void cleanupAllTextures(std::vector<Material>& materials) {
		for (auto& material : materials) {
			int materialType = material.getMaterialType();
			std::cout << "material type" << materialType << "\n";
			if (materialType != 0) {
				
				/*if (material.displacementMap.image != VK_NULL_HANDLE) {
					cleanupTexture(material.displacementMap);
				}*/

				// clean PBR
				if (materialType == 4) {
					if (material.normalMap.image != VK_NULL_HANDLE) {
						cleanupTexture(material.normalMap);
						std::cout << "normal map cleaned." << "\n";
					}
					PBR& pbr = std::get<PBR>(material.materialType);
					if (std::holds_alternative<Texture>(pbr.albedo)) {
						cleanupTexture(std::get<Texture>(pbr.albedo));
					}
					/*if (std::holds_alternative<Texture>(pbr.roughness)) {
						cleanupTexture(std::get<Texture>(pbr.roughness));
					}
					if (std::holds_alternative<Texture>(pbr.metalness)) {
						cleanupTexture(std::get<Texture>(pbr.metalness));
					}*/
				}

				// clean Lambertian
				if (materialType == 1) {
					// clean normal maps (displacement
					if (material.normalMap.image != VK_NULL_HANDLE) {
					cleanupTexture(material.normalMap);
					}
					Lambertian& lambertian = std::get<Lambertian>(material.materialType);
					if (std::holds_alternative<Texture>(lambertian.albedo)) {
						cleanupTexture(std::get<Texture>(lambertian.albedo));
						std::cout << "lambertian material cleaned." << "\n";
					}
				}
			}
		}
	}

	void cleanup() {

		cleanupSwapChain();

		vkDestroySampler(device, textureSampler, nullptr);

		cleanupAllTextures(materials);

		vkDestroyImageView(device, defaultImageView, nullptr);
		vkDestroyImage(device, defaultImage, nullptr);
		vkFreeMemory(device, defaultImageMemory, nullptr);

		vkDestroyImageView(device, cubemapImageView, nullptr);
		vkDestroyImage(device, cubemapImage, nullptr);
		vkFreeMemory(device, cubemapImageMemory, nullptr);

		vkDestroyImageView(device, lambertianCubemapImageView, nullptr);
		vkDestroyImage(device, lambertianCubemapImage, nullptr);
		vkFreeMemory(device, lambertianCubemapImageMemory, nullptr);

		for (size_t i = 0; i < uniformBuffers.size(); i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		for (auto& mesh : sceneGraph.meshes) {
			vkDestroyBuffer(device, mesh.vertexBuffer, nullptr);
			vkFreeMemory(device, mesh.vertexBufferMemory, nullptr);
		}

		for (int i = 0; i < MAX_FRAME_IN_FLIGHT; ++i) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipeline(device, simpleGraphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, swapChainRenderPass, nullptr);

		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);
		glfwDestroyWindow(window);
		glfwTerminate();
	}

	void cleanupHeadless() {
		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		vkDestroyFramebuffer(device, offscreenFramebuffer, nullptr);


		vkDestroyImageView(device, offscreenImageView, nullptr);
		vkFreeMemory(device, offscreenImageMemory, nullptr);
		vkDestroyImage(device, offscreenImage, nullptr);

		for (size_t i = 0; i < uniformBuffers.size(); i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
		}

		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);

		for (auto& mesh : sceneGraph.meshes) {
			vkDestroyBuffer(device, mesh.vertexBuffer, nullptr);
			vkFreeMemory(device, mesh.vertexBufferMemory, nullptr);
		}

		for (int i = 0; i < MAX_FRAME_IN_FLIGHT; ++i) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, commandPool, nullptr);

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyRenderPass(device, offscreenRenderPass, nullptr);


		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
	}

	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "First Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;

		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("create instance failed");
		}

		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

		std::cout << "available extensions:\n";
		for (const auto& extension : extensions) {
			std::cout << "\t" << extension.extensionName << "\n";
		}
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {
				return false;
			}
		}

		return true;
	}

	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}
		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);
		file.seekg(0);
		file.read(buffer.data(), fileSize);
		file.close();

		return buffer;
	}
};

// parse command lines
// argument count = actual count + 1;
// argument vector[0] = file path; argv[argc] = nullptr, the conmmand lines end
CommandLineOptions parseCommandLine(int argc, char* argv[]) {
	CommandLineOptions options;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--headless") {
			if (i + 1 < argc) {
				options.headless = true;
				options.eventFile = argv[++i];
				eventFilePath = options.eventFile;
			}
		}
		else if (arg == "--drawing-size") {
			if (i + 2 < argc) {
				WIDTH = std::atoi(argv[++i]);
				HEIGHT = std::atoi(argv[++i]);
			}
			else {
				std::cerr << "ERROR: '--drawing-size' requires a value." << std::endl;
				exit(1);
			}
		}
		else if (arg == "--scene") {
			if (i + 1 < argc) {
				options.sceneFile = argv[++i];
			}
			else {
				std::cerr << "ERROR: '--scene' requires a file name." << std::endl;
				exit(1);
			}
		}
		else if (arg == "--lambertian") {
			options.lambertian = true;
			if (i + 2 < argc) {
				options.inputFile = argv[++i];
				options.outputFile = argv[++i];
				inputFile = options.inputFile;
				outputFile = options.outputFile;
			}
			else {
				std::cerr << "ERROR: '--lambertian' requires two file names." << std::endl;
				exit(1);
			}
		}
	}

	return options;
}

int main(int argc, char* argv[]) {
	// parse command line options
	CommandLineOptions options = parseCommandLine(argc, argv);

	// load scene graph
	parseScene(sceneGraph, options.sceneFile);

	std::cout << "Headless Mode: " << (options.headless ? "Enabled" : "Disabled") << std::endl;
	std::cout << "Drawing Size: " << options.drawingSize << std::endl;

	FirstTriangle app;
	headless = options.headless;
	lambertian = options.lambertian;



	if (!headless) {
		try {
			// run the app
			app.run();
		}
		catch (const std::exception& message) {
			std::cerr << message.what() << std::endl;
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}
	else {
		try {
			// run the headless mode
			app.runHeadless();
		}
		catch (const std::exception& message) {
			std::cerr << message.what() << std::endl;
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

}

