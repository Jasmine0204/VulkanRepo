#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include <cctype>
#include "camera.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <limits>
#include <algorithm> // For std::find and std::distance
#include <variant> // xor options in materials

#include <glm/gtx/quaternion.hpp>

// This parser is specifically written for .s72 scene files
// I probably used a very dumb way to parse the values
// Also, as progressing, this parser became way more than just a parser...
// 
// Written by Jasmine Chen, andrew id: zitongc
// References: 
// https://kishoreganesh.com/post/writing-a-json-parser-in-cplusplus/

// define structs


struct Mirror {};
struct Simple {};

struct Radiance {
	std::string src;
	std::string type;
	std::string format;
};

struct Environment {
	std::string name;
	Radiance radiance;
};

struct Texture {
	std::string src;
	VkImage image;
	VkDeviceMemory memory;
	VkImageView imageView;
	VkSampler sampler;
};

using ColorOrTexture = std::variant<glm::vec3, Texture>;

struct PBR {
	ColorOrTexture albedo;
	ColorOrTexture roughness;
	ColorOrTexture metalness;
};

struct Lambertian {
	ColorOrTexture albedo;
};

struct Material {
	std::string name;
	Texture normalMap;
	Texture displacementMap;

	// Variant to handle different material types
	std::variant<PBR, Lambertian, Mirror, Environment, Simple, std::monostate> materialType;
	bool isSimple() const {
		return std::holds_alternative<Simple>(materialType);
	}

	// get material type int
	int getMaterialType() const {
		return std::visit([](const auto& arg) -> int {
			using T = std::decay_t<decltype(arg)>;
			if constexpr (std::is_same_v<T, Simple>) return 0; // simple
			else if constexpr (std::is_same_v<T, Lambertian>) return 1; // diffuse
			else if constexpr (std::is_same_v<T, Mirror>) return 2; // mirror
			else if constexpr (std::is_same_v<T, Environment>) return 3; // environment
			else if constexpr (std::is_same_v<T, PBR>) return 4; //  PBR
			else return -1;
			}, materialType);
	}
};



struct Attribute {
	std::string src;
	int offset;
	int stride;
	std::string format;
};

struct Mesh {
	std::string name;
	std::string topology;
	int count;
	std::map<std::string, Attribute> attributes;
	int material;

	Attribute getAttribute(const std::string& attributeName) const {
		auto it = attributes.find(attributeName);
		if (it != attributes.end()) {
			return it->second;
		}
		else {
			throw std::runtime_error("Attribute not found: " + attributeName);
		}
	}
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
};

// I put vertex here just for convenience...
struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec4 tangent;
	glm::vec2 texCoord;
	glm::vec4 color;


	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 5> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 5> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, tangent);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[4].binding = 0;
		attributeDescriptions[4].location = 4;
		attributeDescriptions[4].format = VK_FORMAT_R8G8B8A8_UNORM;
		attributeDescriptions[4].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};

struct SimpleVertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec4 color;


	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(SimpleVertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(SimpleVertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(SimpleVertex, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R8G8B8A8_UNORM;
		attributeDescriptions[2].offset = offsetof(SimpleVertex, color);

		return attributeDescriptions;
	}
};

struct Perspective {
	float aspect;
	float vfov;
	float near;
	float far;
};

struct Camera {
	std::string name;
	Perspective perspective;
};

struct Node {
	std::string name;
	glm::vec3 translation;
	glm::quat rotation;
	glm::vec3 scale;
	int mesh = -1;
	int camera = -1;
	int environment;
	std::vector<int> children;
	bool operator==(const Node& other) const {
		return (name == other.name && translation == other.translation); // compare 'value' to find node
	}
};

glm::mat4 calculateModelMatrix(const Node& node) {
	glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), node.translation);
	glm::mat4 rotationMatrix = glm::toMat4(node.rotation);
	glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), node.scale);

	// model matrix =
	return translationMatrix * rotationMatrix * scaleMatrix;
}

struct Scene {
	std::string name;
	std::vector<int> roots;
};

// animation clip struct
struct AnimationClip {
	std::string type;
	std::string name;
	int node;
	std::string channel;
	std::vector <double> times;
	std::vector<double> values;
	std::string interpolation;
};

// to ensure if a node has animation
struct NodeAnimation {
	std::unordered_map<std::string, AnimationClip> clips; // use channel name as key
};

// stored the parsed values
class SceneGraph {
public:
	std::vector<Mesh> meshes;
	std::vector<Node> nodes;
	std::vector<Camera> cameras;
	std::vector<AnimationClip> clips;
	std::vector<Material> materials;
	Environment environment;
	Scene parsedScene;

	std::unordered_map<int, int> meshIndexMap;
	std::unordered_map<int, int> cameraIndexMap;
	std::unordered_map<int, int> globalNodeIndexMap;
	std::unordered_map<int, int> materialIndexMap;
	std::unordered_map<int, NodeAnimation> nodeAnimations;

	void addMesh(const Mesh& mesh, int globalIndex) {
		int containerIndex = meshes.size();
		meshes.push_back(mesh);
		meshIndexMap[globalIndex] = containerIndex;
	}

	void addNode(const Node& node, int globalPosition) {
		int containerIndex = nodes.size();
		nodes.push_back(node);
		globalNodeIndexMap[globalPosition] = containerIndex;
	}

	void addMaterial(Material& material, int globalPosition) {
		int containerIndex = materials.size();
		materials.push_back(material);
		materialIndexMap[globalPosition] = containerIndex;
	}

	void addCamera(const Camera& camera, int globalPosition) {
		int containerIndex = cameras.size();
		cameras.push_back(camera);
		cameraIndexMap[globalPosition] = containerIndex;
		//std::cout << "In parser, cameras size: " << cameras.size() << "\n";
	}

	void addEnvironment(const Environment& env) {
		environment = env;
	}

	void setScene(const Scene& scene) {
		parsedScene.roots = scene.roots;
		parsedScene = scene;
	}

	void addClip(const AnimationClip& clip) {
		clips.push_back(clip);
	}

	void bindNodeToAnimation(int globalNodeIndex, const AnimationClip& clip) {
		auto it = nodeAnimations.find(globalNodeIndex);
		if (it != nodeAnimations.end()) {
			// if node exist, add clip to this node
			it->second.clips[clip.channel] = clip;
		}
		else {
			//if node doesn't exist, create new node animation and store
			NodeAnimation newAnimation;
			newAnimation.clips[clip.channel] = clip;
			// add new node animation to nodeAnimationMap
			nodeAnimations[globalNodeIndex] = newAnimation;
		}
	}

	int getNodeIndex(const std::vector<Node>& nodes, const Node& node) {
		auto it = std::find(nodes.begin(), nodes.end(), node);
		if (it != nodes.end()) {
			return std::distance(nodes.begin(), it);
		}
		else {
			return -1;
		}
	}

	// give methods that can access the value
	const std::vector<Mesh>& getMeshes() const { return meshes; }
	const std::vector<Node>& getNodes() const { return nodes; }
	std::vector<Material>& getMaterials() { return materials; }
	const std::vector<Camera>& getCameras() const { return cameras; }
	const Environment& getEnvironment() const { return environment; }
	const std::vector<AnimationClip>& getClips() const { return clips; }
	const Scene& getScene() const { return parsedScene; }
};

// main parsing class
class SceneParser {
public:
	SceneGraph sceneGraph;

	SceneParser(const std::string& filename, SceneGraph& sceneGraph) {
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Error opening file\n";
			return;
		}

		std::string line, content;
		while (std::getline(file, line)) {
			content += line + "\n";
		}

		// find "s72-v1"
		const std::string startMarker = "\"s72-v1\"";
		size_t startPos = content.find(startMarker);
		if (startPos == std::string::npos) {
			std::cerr << "Start marker not found in file\n";
			return;
		}

		// calculate start position
		// + startMarker.length() + 1 skip the period after
		size_t pos = startPos + startMarker.length() + 1;
		skipWhitespace(content, pos); // skip blank bytes

		// make sure there is content after to parse
		if (pos >= content.length()) {
			std::cerr << "No content found after start marker\n";
			return;
		}

		skipWhitespace(content, pos);
		// a loop that parse object one by one
		while (pos < content.size()) {
			skipWhitespace(content, pos);

			//parseObject(content, pos);
			if (content[pos] == ',') { // check "," between objects
				pos++;
			}
			if (content[pos] == ']') { // find end mark
				std::cout << "Finish parsing.\n";
				break; // end parsing
			}

			// parse each object seperately
			parseObject(content, pos, sceneGraph);

			skipWhitespace(content, pos);
		}

		skipWhitespace(content, pos);
	}

private:
	int globalPosition = 0;
	// skip all the white spaces, very useful, I forgot to put it between the keys at the begining and caused a lot of errors.
	void skipWhitespace(const std::string& str, size_t& pos) {
		while (pos < str.length() && isspace(str[pos])) {
			pos++;
		}
	}

	// parse object -> parse Mesh/node/camera -> Scene parser loop -> parse object...
	void parseObject(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		//std::vector<int> rootPositions;
		skipWhitespace(str, pos);
		if (str[pos] != '{') {
			std::cerr << "Expected '{' at position " << pos << " in Object start\n";
			return;
		}
		pos++; // skip {

		skipWhitespace(str, pos);

		std::string key = parseString(str, pos);
		skipWhitespace(str, pos);

		if (str[pos] != ':') {
			std::cerr << "Expected ':' after key at position " << pos << "\n";
			return;
		}
		pos++;

		skipWhitespace(str, pos);

		if (key == "type") {

			std::string type = parseString(str, pos);
			std::cout << type << "\n";

			if (str[pos] == ',') {
				pos++;
			} // skip ',' after "MESH"/"NODE"/"CAMERA"...

			if (type == "MESH") {
				globalPosition++;
				Mesh mesh = parseMesh(str, pos, sceneGraph);
			}
			else if (type == "NODE") {
				globalPosition++;
				Node node = parseNode(str, pos, sceneGraph);
			}
			else if (type == "SCENE") {
				globalPosition++;
				Scene scene = parseScene(str, pos, sceneGraph);
			}
			else if (type == "CAMERA") {
				globalPosition++;
				Camera camera = parseCamera(str, pos, sceneGraph);
			}
			else if (type == "DRIVER") {
				globalPosition++;
				AnimationClip clip = parseClip(str, pos, sceneGraph);
			}
			else if (type == "MATERIAL") {
				globalPosition++;
				Material material = parseMaterial(str, pos, sceneGraph);
			}
			else if (type == "ENVIRONMENT") {
				globalPosition++;
				Environment environment = parseEnvironment(str, pos, sceneGraph);
			}
		}
		skipWhitespace(str, pos);

		std::cout << "One object done\n";
		return; // end object

	}

	Environment parseEnvironment(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		Environment env;
		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') {
				pos++; // Skip closing brace
				break;
			}
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key at position " << pos << "\n";
				return env;
			}
			pos++; // Skip colon
			skipWhitespace(str, pos);

			if (key == "name") {
				env.name = parseString(str, pos);
			}
			else if (key == "radiance") {
				env.radiance = parseRadiance(str, pos);
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // Move to next key
			}
		}
		sceneGraph.addEnvironment(env);
		return env;
	}

	Radiance parseRadiance(const std::string& str, size_t& pos) {
		Radiance radiance;
		skipWhitespace(str, pos);
		if (str[pos] != '{') {
			std::cerr << "Expected '{' at position " << pos << "\n";
			return radiance; // Return empty Radiance on error
		}
		pos++; // Skip opening brace

		while (str[pos] != '}') {
			skipWhitespace(str, pos);
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << "\n";
				break;
			}
			pos++; // Skip colon
			skipWhitespace(str, pos);

			if (key == "src") {
				radiance.src = parseString(str, pos);
			}
			else if (key == "type") {
				radiance.type = parseString(str, pos);
			}
			else if (key == "format") {
				radiance.format = parseString(str, pos);
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // Move to next key
			}
		}
		pos++; // Skip closing brace
		return radiance;
	}

	Material parseMaterial(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		Material material;
		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') {
				pos++; // Skip the closing brace
				break;
			}
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" in Material at position " << pos << "\n";
				break;
			}
			pos++; // Skip the colon
			skipWhitespace(str, pos);

			if (key == "name") {
				material.name = parseString(str, pos);
			}
			else if (key == "normalMap") {
				material.normalMap = parseTexture(str, pos);
			}
			else if (key == "displacementMap") {
				material.displacementMap = parseTexture(str, pos);
			}
			else if (key == "pbr") {
				PBR pbrMaterial;
				pbrMaterial.albedo = parseColorOrTexture(str, pos);
				pbrMaterial.roughness = parseColorOrTexture(str, pos);
				pbrMaterial.metalness = parseColorOrTexture(str, pos);
				material.materialType = pbrMaterial;
			}
			else if (key == "lambertian") {
				Lambertian lambertianMaterial;
				skipWhitespace(str, pos);

				if (str[pos] != '{') {
					std::cerr << "Expected '{' at position " << pos << "\n";
					return material; // Return empty Radiance on error
				}
				pos++;

				while (true) {
					skipWhitespace(str, pos);
					if (str[pos] == '}') {
						pos++; // Skip the closing brace
						break;
					}
					std::string key = parseString(str, pos);
					skipWhitespace(str, pos);
					if (str[pos] != ':') {
						std::cerr << "Expected ':' after key \"" << key << "\" in lambertian at position " << pos << "\n";
						break;
					}
					pos++; // Skip colon
					skipWhitespace(str, pos);
					if (key == "albedo") { lambertianMaterial.albedo = parseColorOrTexture(str, pos); }
					material.materialType = lambertianMaterial;
				}


			}
			else if (key == "mirror") {
				material.materialType = Mirror{};
				// skip the blank brackets
				pos++;
				skipWhitespace(str, pos);
				pos++;
			}
			else if (key == "environment") {
				material.materialType = Environment{};
				// skip the blank brackets
				pos++;
				skipWhitespace(str, pos);
				pos++;
			}
			else if (key == "simple") {
				material.materialType = Simple{};
				// skip the blank brackets
				pos++;
				skipWhitespace(str, pos);
				pos++;
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // Move to the next key
			}
		}
		sceneGraph.addMaterial(material, globalPosition);
		return material;
	}

	ColorOrTexture parseColorOrTexture(const std::string& str, size_t& pos) {
		skipWhitespace(str, pos);
		if (str[pos] == '[') {
			// Parse as Color
			return parseVec3(str, pos);
		}
		else if (str[pos] == '{') {
			// Parse as TextureSource
			return parseTexture(str, pos);
		}
		else {
			std::cerr << "Unexpected character for ColorOrTexture at position " << pos << ": " << str[pos] << "\n";
			return glm::vec3(); // Returning empty variant on error
		}
	}

	Texture parseTexture(const std::string& str, size_t& pos) {
		Texture texture;
		skipWhitespace(str, pos);
		if (str[pos] != '{') {
			std::cerr << "Expected '{' at position " << pos << "\n";
			return texture; // Returning an empty TextureSource on error
		}
		pos++; // Skip the opening brace

		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') {
				pos++; // Skip the closing brace
				break;
			}
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << "\n";
				break;
			}
			pos++; // Skip the colon
			skipWhitespace(str, pos);

			if (key == "src") {
				texture.src = parseString(str, pos);
			}

			skipWhitespace(str, pos);
		}
		return texture;
	}

	AnimationClip parseClip(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		AnimationClip clip;
		skipWhitespace(str, pos);

		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') {
				pos++; // skip the last }
				//std::cout << "One mesh done\n";
				break;
			}
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << " in animation clip\n";
				break;
			}
			pos++; // skip :
			skipWhitespace(str, pos);

			if (key == "name") {
				clip.name = parseString(str, pos);
			}
			else if (key == "node") {
				clip.node = parseInt(str, pos);
			}
			else if (key == "channel") {
				clip.channel = parseString(str, pos);
			}
			else if (key == "times") {
				clip.times = parseDoubleArray(str, pos);
			}
			else if (key == "values") {
				clip.values = parseDoubleArray(str, pos);
			}
			else if (key == "interpolation") {
				clip.interpolation = parseString(str, pos);
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // next key
			}
		}

		sceneGraph.bindNodeToAnimation(clip.node, clip);
		sceneGraph.addClip(clip);
		return clip;
	}

	std::string parseString(const std::string& str, size_t& pos) {
		std::string result;
		skipWhitespace(str, pos);
		if (str[pos] != '"') {
			std::cerr << "Expected '\"' at position " << pos << " in String\n";
			return "";
		}
		pos++; // skip start "
		while (pos < str.size() && str[pos] != '"') { // until get another "
			result += str[pos];
			pos++;
		}
		if (pos < str.size() && str[pos] == '"') {
			pos++; // skip the end "
		}
		else {
			// print error if no end "
			std::cerr << "Expected '\"' at the end of the string at position " << pos << "\n";
		}
		return result;
	}

	Mesh parseMesh(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		Mesh mesh;
		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') {
				pos++; // skip the last }
				//std::cout << "One mesh done\n";
				break;
			}
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << "\n";
				break;
			}
			pos++; // skip :
			skipWhitespace(str, pos);

			if (key == "name") {
				mesh.name = parseString(str, pos);
			}
			else if (key == "topology") {
				mesh.topology = parseString(str, pos);
			}
			else if (key == "count") {
				mesh.count = parseInt(str, pos);
			}
			else if (key == "attributes") {
				parseAttributes(mesh.attributes, str, pos);
			}
			else if (key == "material") {
				mesh.material = parseInt(str, pos);
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // next key
			}
		}
		sceneGraph.addMesh(mesh, globalPosition);
		return mesh;
	}

	Node parseNode(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		Node node;
		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') { // if end?
				pos++; // skip "}"
				break;
			}
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key at position " << pos << "\n";
				return node;
			}
			pos++; // skip :
			skipWhitespace(str, pos);

			if (key == "name") {
				node.name = parseString(str, pos);
			}
			else if (key == "translation") {
				node.translation = parseVec3(str, pos);
			}
			else if (key == "rotation") {
				glm::vec4 vec = parseVec4(str, pos);
				node.rotation = glm::quat(vec.w, vec.x, vec.y, vec.z);;
			}
			else if (key == "scale") {
				node.scale = parseVec3(str, pos);
			}
			else if (key == "mesh") {
				node.mesh = parseInt(str, pos);
			}
			else if (key == "camera") {
				node.camera = parseInt(str, pos);
			}
			else if (key == "children") {
				node.children = parseIntArray(str, pos);
			}
			else if (key == "environment") {
				node.environment = parseInt(str, pos);
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // skip , to next key
			}
		}

		sceneGraph.addNode(node, globalPosition);

		return node;
	}

	Scene parseScene(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		Scene scene;

		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') {
				pos++; // skip  '}'
				std::cout << "Scene is done.\n";
				break;
			}
			std::string key = parseString(str, pos);

			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << "\n";
				break;
			}
			pos++; // skip ':'
			skipWhitespace(str, pos);

			if (key == "name") {
				scene.name = parseString(str, pos);
			}
			else if (key == "roots") {
				scene.roots = parseIntArray(str, pos);
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // skip ','
				skipWhitespace(str, pos);
			}
		}

		sceneGraph.setScene(scene);
		return scene;
	}

	Camera parseCamera(const std::string& str, size_t& pos, SceneGraph& sceneGraph) {
		Camera camera;

		while (true) {
			skipWhitespace(str, pos);

			if (str[pos] == '}') { // if end
				pos++; // skip '}'
				break;
			}

			// get key
			std::string key = parseString(str, pos);


			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << "\n";
				break;
			}
			pos++; // skip ':'
			skipWhitespace(str, pos);

			// compare key to value
			if (key == "name") {
				camera.name = parseString(str, pos);
			}
			else if (key == "perspective") {
				camera.perspective = parsePerspective(str, pos);
			}

			skipWhitespace(str, pos);

			if (str[pos] == ',') {
				pos++; // next key
			}
		}
		sceneGraph.addCamera(camera, globalPosition);
		return camera;
	}

	// helper parser functions, mainly deal different kinds of number values
	glm::vec3 parseVec3(const std::string& str, size_t& pos) {
		glm::vec3 vector3{ 0.0f,0.0f,0.0f };
		skipWhitespace(str, pos);
		if (str[pos] != '[') {
			std::cerr << "Expected '[' at position " << pos << "\n";
			return vector3;
		}
		++pos;

		vector3.x = parseDouble(str, pos);
		++pos;
		vector3.y = parseDouble(str, pos);
		++pos;
		vector3.z = parseDouble(str, pos);

		skipWhitespace(str, pos);
		if (str[pos] != ']') {
			std::cerr << "Expected ']' at position " << pos << "\n";
			return vector3;
		}
		++pos;
		return vector3;
	}

	glm::vec4 parseVec4(const std::string& str, size_t& pos) {
		glm::vec4 vector4{ 0.0f,0.0f,0.0f,0.0f };
		skipWhitespace(str, pos);
		if (str[pos] != '[') {
			std::cerr << "Expected '[' at position " << pos << "\n";
			return vector4;
		}
		++pos; // skip '['

		vector4.x = parseDouble(str, pos);
		++pos;
		vector4.y = parseDouble(str, pos);
		++pos;
		vector4.z = parseDouble(str, pos);
		++pos;
		vector4.w = parseDouble(str, pos);

		skipWhitespace(str, pos);
		if (str[pos] != ']') {
			std::cerr << "Expected ']' at position " << pos << "\n";
			return vector4;
		}
		++pos;
		return vector4;
	}

	double parseDouble(const std::string& str, size_t& pos) {
		skipWhitespace(str, pos);
		size_t start = pos;
		// be sure to process exponent
		bool hasExponent = false;

		while (pos < str.size()) {
			char ch = str[pos];
			if (isdigit(ch) || ch == '.' || (ch == '-' && pos == start) || (ch == '+' && pos == start)) {
				// for these parts, let it go
				++pos;
			}
			else if ((ch == 'e' || ch == 'E') && !hasExponent) {
				// get exponent
				hasExponent = true;
				++pos;
				if (pos < str.size() && (str[pos] == '+' || str[pos] == '-')) {
					++pos; // also include signs
				}
				// make sure there are numbers after e
				if (pos == str.size() || !isdigit(str[pos])) {
					throw std::invalid_argument("Invalid scientific notation: missing exponent number.");
				}
			}
			else {
				break;
			}
		}
		double result = std::stod(str.substr(start, pos - start));
		return result;
	}

	Perspective parsePerspective(const std::string& str, size_t& pos) {
		Perspective perspective{ 1.778,0.5,0.1,100 };
		skipWhitespace(str, pos);
		if (str[pos] != '{') {
			std::cerr << "Expected '{' at position " << pos << "\n";
			return perspective;
		}
		pos++; // skip start '{'

		while (true) {
			skipWhitespace(str, pos);

			if (str[pos] == '}') { // if reach the end of perspective
				pos++; // skip '}'
				break;
			}

			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);

			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << "\n";
				break;
			}
			pos++; // skip ':'
			skipWhitespace(str, pos);

			if (key == "aspect") {
				perspective.aspect = parseDouble(str, pos);
			}
			else if (key == "vfov") {
				perspective.vfov = parseDouble(str, pos);
			}
			else if (key == "near") {
				perspective.near = parseDouble(str, pos);
			}
			else if (key == "far") {
				perspective.far = parseDouble(str, pos);
			}

			skipWhitespace(str, pos);

			if (str[pos] == ',') {
				pos++; // to next key
			}
		}

		return perspective;
	}

	Attribute parseAttribute(const std::string& str, size_t& pos) {
		Attribute attr;
		skipWhitespace(str, pos);
		if (str[pos] == '{') {
			pos++;// skip { after "POSITION":
		}

		while (str[pos] != '}') {
			std::string key = parseString(str, pos);
			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after key \"" << key << "\" at position " << pos << "\n";
				break;
			}
			pos++;
			skipWhitespace(str, pos);

			if (key == "src") {
				attr.src = parseString(str, pos);
			}
			else if (key == "offset") {
				attr.offset = parseInt(str, pos);
			}
			else if (key == "stride") {
				attr.stride = parseInt(str, pos);
			}
			else if (key == "format") {
				attr.format = parseString(str, pos);
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // skip , to next key
			}
			skipWhitespace(str, pos);
		}
		pos++; // skip end }
		return attr;
	}

	int parseInt(const std::string& str, size_t& pos) {
		skipWhitespace(str, pos);
		std::string numberStr;
		while (pos < str.length() && isdigit(str[pos])) {
			numberStr += str[pos];
			pos++;
		}
		if (str[pos] == ',') {
			pos++;
		}
		return std::stoi(numberStr);
	}

	void parseAttributes(std::map<std::string, Attribute>& attributes, const std::string& str, size_t& pos) {
		skipWhitespace(str, pos);
		if (str[pos] != '{') {
			std::cerr << "Expected '{' at position " << pos << "\n";
			return;
		}
		pos++; // skip { before "POSITION"

		while (true) {
			skipWhitespace(str, pos);
			if (str[pos] == '}') {
				pos++; // end attributes
				break;
			}

			std::string attrName = parseString(str, pos);

			skipWhitespace(str, pos);
			if (str[pos] != ':') {
				std::cerr << "Expected ':' after attribute name \"" << attrName << "\" at position " << pos << "\n";
				break;
			}
			pos++; // skip colon

			Attribute attribute = parseAttribute(str, pos);
			attributes[attrName] = attribute;

			skipWhitespace(str, pos);

			if (str[pos] == ',') {
				pos++; // parse next attribute
			}
		}
	}

	std::vector<int> parseIntArray(const std::string& str, size_t& pos) {
		std::vector<int> array;
		skipWhitespace(str, pos);
		if (str[pos] != '[') {
			std::cerr << "Expected '[' at position " << pos << "\n";
			return array;
		}
		pos++; // skip start'['

		while (str[pos] != ']') {
			skipWhitespace(str, pos);
			std::string number = "";
			while (pos < str.size() && (isdigit(str[pos]) || str[pos] == '-')) {
				number += str[pos++];
			}
			if (!number.empty()) {
				array.push_back(std::stoi(number));
			}

			skipWhitespace(str, pos);
			if (str[pos] == ',') {
				pos++; // skip ',' to next value
			}
		}
		pos++; // skip end ']'

		return array;
	}

	std::vector<double> parseDoubleArray(const std::string& str, size_t& pos) {
		std::vector<double> result;
		skipWhitespace(str, pos);
		if (str[pos] != '[') {
			std::cerr << "Expected '[' at position " << pos << "\n";
			return result;
		}
		pos++;

		while (pos < str.size() && str[pos] != ']') {
			skipWhitespace(str, pos);
			size_t start = pos;
			bool hasExponent = false;

			while (pos < str.size()) {
				char ch = str[pos];
				if (isdigit(ch) || ch == '.' || (ch == '-' && pos == start) || (ch == '+' && pos == start)) {
					++pos;
				}
				else if ((ch == 'e' || ch == 'E') && !hasExponent) {
					hasExponent = true;
					++pos;
					if (pos < str.size() && (str[pos] == '+' || str[pos] == '-')) {
						++pos;
					}
					if (pos == str.size() || !isdigit(str[pos])) {
						throw std::invalid_argument("Invalid scientific notation: missing exponent number.");
					}
				}
				else {
					break;
				}
			}

			if (start != pos) {
				double value = std::stod(str.substr(start, pos - start));
				result.push_back(value);
			}

			// skip ','
			if (str[pos] == ',') {
				pos++;
			}

			skipWhitespace(str, pos);
		}
		if (str[pos] == ',') {
			pos++; //
		}
		pos++;
		return result;
	}
};

int parseScene(SceneGraph& sceneGraph, const std::string& filename) {
	SceneParser parser(filename, sceneGraph);
	return 0;
}