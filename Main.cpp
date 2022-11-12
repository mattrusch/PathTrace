// Main.cpp

#include "stdafx.h"
#include "PathTrace.h"
#include "Win32Application.h"

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
	PathTrace pathTrace(2560, 1600, L"PathTrace");
	return Win32Application::Run(&pathTrace, hInstance, nCmdShow);
}
