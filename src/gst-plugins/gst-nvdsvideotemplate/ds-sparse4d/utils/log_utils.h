/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOG_UTILS_H
#define LOG_UTILS_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <map>
#include <memory>
#include <algorithm>
#include <stack>

// ANSI color codes
#define ANSI_COLOR_ERROR     "\x1b[31m"      // Bright red - standard for errors
#define ANSI_COLOR_SUCCESS   "\x1b[38;2;118;185;0m"  // NVIDIA green (76B900 hex)
#define ANSI_COLOR_WARNING   "\x1b[33m"      // Yellow/amber - standard for warnings
#define ANSI_COLOR_INFO      "\x1b[38;2;0;175;255m"  // Cyan - better for info than blue
#define ANSI_COLOR_INFO2     "\x1b[38;2;147;161;161m"   // Soft gray
#define ANSI_COLOR_FRAME     "\x1b[38;2;255;165;0m"     // Orange for frame
#define ANSI_COLOR_RESET     "\x1b[0m"

/**
 * @brief Enumeration for logging levels
 *
 * Defines the severity levels for logging. Higher levels include all lower levels.
 * - INFO: Display all messages (info, warnings, errors)
 * - WARN: Display warnings and errors only
 * - ERROR: Display errors only
 * Note: persistent_info messages are always displayed regardless of log level
 */
enum class LogLevel {
    INFO,   // Display all: info, warn, error
    WARN,   // Display: warn, error
    ERROR   // Display: error only
};

class PrettyLogger {
private:
    using high_res_clock = std::chrono::high_resolution_clock;
    using time_point = std::chrono::time_point<high_res_clock>;

    // Special message types that need special formatting
    enum class SpecialMessageType {
        NONE,
        INPUT_TENSORS,
        DYNAMIC_TENSOR
    };

    // Process node for building a tree structure
    struct ProcessNode {
        std::string name;
        int frame_num;
        time_point start_time;
        std::vector<std::string> info_messages;
        bool is_main_process;
        std::vector<std::shared_ptr<ProcessNode>> children;
        ProcessNode* parent;

        ProcessNode(const std::string& n, int frame, bool main, ProcessNode* p)
            : name(n), frame_num(frame), is_main_process(main), parent(p) {}
    };

    // Member variables
    LogLevel log_level;
    int current_frame;
    std::shared_ptr<ProcessNode> root_process;
    std::stack<ProcessNode*> process_stack;
    std::unordered_map<std::string, ProcessNode*> process_map;

    // Track whether we need to add spacing before the next process
    bool needsSpacingForNextProcess;

    // Track if the last message was a root-level info
    bool lastMessageWasRootLevelInfo;

    // Helper to get unique process key
    std::string getProcessKey(const std::string& process, int frameNum) const {
        return process + "_" + std::to_string(frameNum);
    }

    // Helper to format time with appropriate units
    std::string formatTime(double ms_duration) const {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << std::setw(7) << ms_duration << "ms";
        return ss.str();
    }

    // Detect special message type
    SpecialMessageType detectSpecialMessageType(const std::string& message) const {
        // Case-sensitive check for exact "Input tensors:" string at the beginning
        if (message.compare(0, 14, "Input tensors:") == 0) {
            return SpecialMessageType::INPUT_TENSORS;
        }
        // Case-sensitive check for Dynamic tensor lines
        if (message.compare(0, 15, "Dynamic tensor:") == 0) {
            return SpecialMessageType::DYNAMIC_TENSOR;
        }
        // Treat info messages that should be at root level as special
        if (needsSpacingForNextProcess || process_stack.empty()) {
            return SpecialMessageType::INPUT_TENSORS;  // Use same formatting as input tensors
        }
        return SpecialMessageType::NONE;
    }

    // Get indentation based on context and message type
    std::string getIndentationForMessage(const std::string& message, bool isInfoLine = true) const {
        SpecialMessageType messageType = detectSpecialMessageType(message);

        // Default indentation for regular messages
        if (messageType == SpecialMessageType::NONE) {
            if (!process_stack.empty()) {
                return getTreeIndentation(process_stack.top(), isInfoLine);
            } else {
                return "   │    └─ ";  // Default when no context
            }
        }

        // Special handling for INPUT_TENSORS messages
        if (messageType == SpecialMessageType::INPUT_TENSORS) {
            // Input tensors should appear at the root level with specific indentation
            return "   └─ ";  // Special indentation for Input tensors
        }


        // Special handling for DYNAMIC_TENSOR messages
        if (messageType == SpecialMessageType::DYNAMIC_TENSOR) {
            // Dynamic tensors should be indented consistently with other info messages
            // They should appear at same level as info messages, but after a special message
            if (!process_stack.empty()) {
                return getTreeIndentation(process_stack.top(), true);
            } else {
                return "   │    └─ ";
            }
        }

        // Default fallback
        return "   │    └─ ";
    }

    // Get tree-style indentation for any process
    std::string getTreeIndentation(ProcessNode* node, bool isInfoLine = true) const {
        if (!node) return "   │";

        std::vector<ProcessNode*> path;
        ProcessNode* current = node;

        // Build path from root to node
        while (current && !current->is_main_process) {
            path.push_back(current);
            current = current->parent;
        }

        // Reverse to get root-to-node order
        std::reverse(path.begin(), path.end());

        // Base indentation
        std::string indent = "   │";

        // Add appropriate tree branches based on path
        if (path.size() > 0) {
            for (size_t i = 0; i < path.size() - 1; i++) {
                indent += "    │";
            }

            // Last segment depends on whether this is an info line or a process line
            if (isInfoLine) {
                indent += "    └─ ";
            } else {
                indent += "    ├─► ";
            }
        } else {
            // For top-level processes (direct children of main process)
            indent += isInfoLine ? "    └─ " : "    ├─► ";
        }

        return indent;
    }

    // Get indentation for completed process message
    std::string getCompletionIndentation(ProcessNode* node) const {
        if (!node) return "   │";
        if (node->is_main_process) return "";

        return getTreeIndentation(node, false);
    }

    // Get spacing indentation for separator lines
    std::string getSpacingIndentation(ProcessNode* node) const {
        if (!node) return "   │";

        std::vector<ProcessNode*> path;
        ProcessNode* current = node;

        // Build path from root to node
        while (current && !current->is_main_process) {
            path.push_back(current);
            current = current->parent;
        }

        // Reverse to get root-to-node order
        std::reverse(path.begin(), path.end());

        // Base indentation
        std::string indent = "   │";

        // Add appropriate tree branches based on path
        for (size_t i = 0; i < path.size(); i++) {
            indent += "    │";
        }

        return indent;
    }

    /**
     * @brief Convert string to LogLevel enum
     *
     * @param[in] level_str String representation of log level
     * @return LogLevel enum value
     */
    LogLevel stringToLogLevel(const std::string& level_str) const {
        std::string lower_str = level_str;
        std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
        
        if (lower_str == "info" || lower_str == "information") {
            return LogLevel::INFO;
        } else if (lower_str == "warn" || lower_str == "warning") {
            return LogLevel::WARN;
        } else {
            return LogLevel::ERROR;  // Default to ERROR
        }
    }

public:
    PrettyLogger(LogLevel level = LogLevel::ERROR) 
        : log_level(level),
          current_frame(-1),
          root_process(nullptr),
          needsSpacingForNextProcess(false),
          lastMessageWasRootLevelInfo(false)
    {}

    /**
     * @brief Set the logging level
     *
     * @param[in] level_str String representation of log level (info, warn, error)
     */
    void setLogLevel(const std::string& level_str) {
        log_level = stringToLogLevel(level_str);
    }

    /**
     * @brief Set the logging level using enum
     *
     * @param[in] level LogLevel enum value
     */
    void setLogLevel(LogLevel level) {
        log_level = level;
    }

    void reset() {
        root_process = nullptr;
        process_stack = std::stack<ProcessNode*>();
        process_map.clear();
        current_frame = -1;
        needsSpacingForNextProcess = false;
        lastMessageWasRootLevelInfo = false;
    }

    /**
     * @brief Display a warning message
     *
     * Warnings are displayed when log level is WARN or INFO.
     *
     * @param[in] message The warning message to display
     */
    void warning(const std::string& message) {
        // Display warnings only if log level is WARN or INFO
        if (log_level == LogLevel::ERROR) {
            return;
        }

        // Indentation only used in INFO mode
        if (log_level == LogLevel::INFO) {
            std::string indent;
            if (!process_stack.empty()) {
                indent = getTreeIndentation(process_stack.top(), true);
            } else {
                indent = "   │    └─ ";
            }
            std::cout << indent;
        }

        std::cout << ANSI_COLOR_WARNING << "⚠️  " << message << ANSI_COLOR_RESET << "\n";
    }

    /**
     * @brief Display an error message
     *
     * Errors are always displayed regardless of log level.
     *
     * @param[in] message The error message to display
     */
    void error(const std::string& message) {
        // Indentation only used in INFO mode
        if (log_level == LogLevel::INFO) {
            // Get indentation based on current process context
            std::string indent;
            if (!process_stack.empty()) {
                indent = getTreeIndentation(process_stack.top(), true);
            } else {
                indent = "   │    └─ ";
            }
            std::cout << indent;
        }

        std::cout << ANSI_COLOR_ERROR << "❌ " << message << ANSI_COLOR_RESET << "\n";
    }

    /**
     * @brief Display an informational message
     *
     * Info messages are displayed only when log level is INFO.
     *
     * @param[in] message The informational message to display
     */
    void info(const std::string& message) {
        // Display info only if log level is INFO
        if (log_level != LogLevel::INFO) {
            return;
        }

        // Get appropriate indentation based on message type and context
        SpecialMessageType messageType = detectSpecialMessageType(message);
        std::string indent = getIndentationForMessage(message, true);

        // Store info message in current process if available
        if (!process_stack.empty() && messageType == SpecialMessageType::NONE) {
            process_stack.top()->info_messages.push_back(message);
        }

        // Print the message with appropriate formatting
        std::cout << indent
                 << ANSI_COLOR_INFO2 << message << ANSI_COLOR_RESET << "\n";

        // Reset spacing flag if this was a top-level message
        if (messageType == SpecialMessageType::INPUT_TENSORS) {
            needsSpacingForNextProcess = false;
        }

        // Set lastMessageWasRootLevelInfo to true if this was a root-level info message
        lastMessageWasRootLevelInfo = messageType == SpecialMessageType::INPUT_TENSORS;
    }

    /**
     * @brief Prints a persistent information message that is always displayed
     *
     * This function provides critical messages that should always be visible regardless of the 
     * logger's log level. When log level is INFO, it uses proper formatting. Otherwise, it 
     * prints directly without formatting.
     *
     * @param[in] message The message to be displayed
     */
    void persistent_info(const std::string& message) {
        if (log_level != LogLevel::INFO) {
            // In ERROR or WARN mode, print directly to console without formatting
            std::cout << message << "\n";
            return;
        }

        // If log level is not INFO, use the regular info method to maintain consistent formatting
        info(message);
    }

    /**
     * @brief Mark the start of a process for timing and logging
     *
     * Process timing is displayed only when log level is INFO.
     *
     * @param[in] process The name of the process
     * @param[in] frameNum The frame number (-1 for non-frame-specific processes)
     * @param[in] isMainProcess Whether this is the main process
     * @param[in] additionalInfo Optional additional information to display
     */
    void processStart(const std::string& process, int frameNum,
                      bool isMainProcess = false,
                      const std::string& additionalInfo = "") {
        // Process timing is only displayed in INFO mode
        if (log_level != LogLevel::INFO) {
            return;
        }

        // Check for frame change and reset state if needed
        if (frameNum != current_frame) {
            current_frame = frameNum;
            root_process = nullptr;
            process_stack = std::stack<ProcessNode*>();
            process_map.clear();
            needsSpacingForNextProcess = false;
            lastMessageWasRootLevelInfo = false;
        }

        std::string process_key = getProcessKey(process, frameNum);

        // Create new process node
        ProcessNode* parent = nullptr;
        if (!isMainProcess && !process_stack.empty()) {
            parent = process_stack.top();
        }

        auto node = std::make_shared<ProcessNode>(process, frameNum, isMainProcess, parent);
        node->start_time = high_res_clock::now();

        // Add to parent's children if not main process
        if (parent) {
            parent->children.push_back(node);
        } else if (isMainProcess) {
            // Store main process as root
            root_process = node;
        }

        // Add to process map and stack
        ProcessNode* node_ptr = node.get();
        process_map[process_key] = node_ptr;
        process_stack.push(node_ptr);

        // Format and print start message
        if (isMainProcess) {
            // Main process uses fixed formatting
            if (frameNum >= 0) {
                std::cout << "\n" << process << " Frame-" << frameNum
                        << " │ ⚡ Starting" << "\n"
                        << "   │    │\n";
            } else {
                std::cout << "\n" << process
                        << " │ ⚡ Starting" << "\n"
                        << "   │    │\n";
            }
            // Reset spacing flag since main process adds its own spacing
            needsSpacingForNextProcess = false;
            lastMessageWasRootLevelInfo = false;
        } else {
            // If not main process and spacing is needed, add it before the process line
            if (needsSpacingForNextProcess && parent) {
                std::string spacing = getSpacingIndentation(parent);
                std::cout << spacing << "    │\n";
            }

            // Always add a vertical spacing line before a new process starts
            // This guarantees visual separation between info messages and process blocks
            if (!isMainProcess && parent && process_stack.size() > 1) {
                std::string spacing = getSpacingIndentation(parent);
                std::cout << spacing << "    │\n";
            }

            // If the last message was a root-level info and this is a first-level process,
            // add a line break for visual separation
            if (lastMessageWasRootLevelInfo && process_stack.size() == 1) {
                std::cout << "   │    │\n";
            }

            // Get tree-style indentation for child process
            std::string indentation = getTreeIndentation(node_ptr, false);
            std::cout << indentation << process << " │ ⚡ Starting" << "\n";

            // Reset flags
            needsSpacingForNextProcess = false;
            lastMessageWasRootLevelInfo = false;
        }

        // Handle additional info
        if (!additionalInfo.empty()) {
            node->info_messages.push_back(additionalInfo);
            std::string info_indent = getTreeIndentation(node_ptr, true);
            std::cout << info_indent
                     << ANSI_COLOR_INFO2 << additionalInfo << ANSI_COLOR_RESET << "\n";
        }
    }

    /**
     * @brief Mark the completion of a process and display timing information
     *
     * Process completion is displayed only when log level is INFO.
     *
     * @param[in] process The name of the process
     * @param[in] frameNum The frame number
     * @param[in] success Whether the process completed successfully
     * @param[in] isMainProcess Whether this is the main process
     * @param[in] additionalInfo Optional additional information to display
     */
    void processComplete(const std::string& process, int frameNum,
                         bool success = true,
                         bool isMainProcess = false,
                         const std::string& additionalInfo = "") {
        // Process completion is only displayed in INFO mode
        if (log_level != LogLevel::INFO) {
            return;
        }

        std::string process_key = getProcessKey(process, frameNum);
        auto it = process_map.find(process_key);

        if (it == process_map.end()) {
            // Process not found
            return;
        }

        ProcessNode* node = it->second;

        // Calculate timing
        auto end_time = high_res_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - node->start_time);
        double ms_duration = duration.count() / 1000.0;

        // Format timing info
        std::string timing_info = formatTime(ms_duration);

        // Create success and failure symbols with appropriate colors
        // Using more prominent characters for better visibility
        std::string successSymbol = std::string(ANSI_COLOR_SUCCESS) + "✔" + std::string(ANSI_COLOR_RESET);
        std::string failureSymbol = std::string(ANSI_COLOR_ERROR) + "✘" + std::string(ANSI_COLOR_RESET);

        // Format and print completion message
        if (isMainProcess) {
            // Main process uses fixed formatting
            if (frameNum >= 0) {
                std::cout << process << " Frame-" << frameNum
                        << " │ ⏱ " << std::right << timing_info
                        << " │ " << (success ? successSymbol : failureSymbol) << " "
                        << (success ? "Completed" : "Failed") << "\n";
            } else {
                std::cout << process
                        << " │ ⏱ " << std::right << timing_info
                        << " │ " << (success ? successSymbol : failureSymbol) << " "
                        << (success ? "Completed" : "Failed") << "\n";
            }
        } else {
            // Get indentation for this process in the tree
            std::string indentation = getCompletionIndentation(node);

            // Output completion message
            std::cout << indentation << process
                     << " │ ⏱ " << std::right << timing_info
                     << " │ " << (success ? successSymbol : failureSymbol) << " "
                     << (success ? "Completed" : "Failed") << "\n";

            // Set spacing flag for next process that starts after this completion
            needsSpacingForNextProcess = true;

            // Add a newline with proper indentation after completion for better visual separation
            if (node->parent) {
                // Use the parent's spacing indentation to get the right tree structure
                std::string spacing = getSpacingIndentation(node->parent);
                std::cout << spacing << "    │\n";
            }
        }

        // Handle additional info
        if (!additionalInfo.empty()) {
            std::string info_indent = getTreeIndentation(node, true);
            std::cout << info_indent
                     << ANSI_COLOR_INFO2 << additionalInfo << ANSI_COLOR_RESET << "\n";
        }

        // Pop from process stack if it's the current process
        if (!process_stack.empty() && process_stack.top() == node) {
            process_stack.pop();
        }

        // Remove from process map
        process_map.erase(process_key);
    }

};

#endif /* LOG_UTILS_H */