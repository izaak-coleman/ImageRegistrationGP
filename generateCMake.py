#!/usr/bin/python

from optparse import OptionParser
import re
import sys


def cmaketext(project, executable, src):
    cmake = [
		    "cmake_minimum_required(VERSION 2.8)",
				"project( " + project + " )",
				"# ----- Linking to openCV 3.1.0 -----",
				"#THE FOLLOWING CONFIGURATION REQUIRED::",
				"set(OpenCV_DIR \"/vol/bitbucket/ic711/opencv-3.1.0/release\")",
				"set(OpenCV_CONFIG_PATH \"/vol/bitbucket/ic711/opencv-3.1.0/cmake\")",
				"find_package( OpenCV 3.1.0 EXACT CONFIG REQUIRED )  #FINDS package config for OpenCV 3.1.0",
				"message(\"OpenCV package version \" ${OpenCV_VERSION})",
				"# ----- link libraries -----",
				"add_executable( " + executable + " \"" + src + "\" )",
				"target_link_libraries( " + executable + " ${OpenCV_LIBS})"
    ]

    return cmake


def main():
    parser = OptionParser("usage :%prog <project_name> <executable> <src>")

    (options, args) = parser.parse_args()
    if len(args) != 3:
		    parser.error("Incorrect number of arguments")




    cmake = cmaketext(args[0], args[1], args[2])

    for line in cmake:
        sys.stdout.write(line)
        sys.stdout.write("\n")

main()



