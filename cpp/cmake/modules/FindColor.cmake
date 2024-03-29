cmake_minimum_required(VERSION 3.7.2)

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColorReset "${Esc}[m")
  set(Red     "${Esc}[1;31m")
  set(Green   "${Esc}[1;32m")
  set(Yellow  "${Esc}[1;33m")
  set(Blue    "${Esc}[1;34m")
  set(Magenta "${Esc}[1;35m")
  set(Cyan    "${Esc}[1;36m")
  set(White   "${Esc}[1;37m")
endif()