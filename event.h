#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

enum class EventType {
    AVAILABLE,
    PLAY,
    SAVE,
    MARK,
};

// event struct
struct Event {
    long long timestamp;
    EventType type;
    std::vector<std::string> params;
};

EventType parseEventType(const std::string& eventType) {
    if (eventType == "AVAILABLE") {

        return EventType::AVAILABLE;
    }
    else if (eventType == "PLAY") {

        return EventType::PLAY;
    }
    else if (eventType == "SAVE") {

        return EventType::SAVE;
    }
    else if (eventType == "MARK") {
        return EventType::MARK;
    }
    else {
        throw std::invalid_argument("Unknown event type: " + eventType);
    }
}

std::vector<Event> parseEventFile(const std::string& filePath) {
    std::vector<Event> events;

    std::ifstream file(filePath);
    if (!file) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return events;
    }
    std::string line;


    while (getline(file, line)) {

        std::istringstream iss(line);
        long long ts;
        std::string eventType;
        iss >> ts >> eventType;

        Event event;
        event.timestamp = ts;
        event.type = parseEventType(eventType); // convert string to EventType
        std::string param;
        while (iss >> param) {
            event.params.push_back(param);
        }

        events.push_back(event);
    }

    return events;
}

