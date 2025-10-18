# mmcp
## An MCP server for Multimodal Input

This is a prototype stage demonstration of a multimodal MCP server that currently has three modality inputs -
1. Text
2. Image
3. Audio
4. Video modality with audio support

#### The first one is just simple but let's discuss the next three in detail:-


## 1. Image Modality
Here we are doing two things
1. Reading text from image if any
2. Captioning the image using an opensource model


## 2. Audio Modality
Similarly here we're doing
1. Converting audio to text if any
2. Using an open source model to characterize or classify the background noise

## 3. Video Modality
Here the image is split into frames every tenth of a second each of those frames are passed to a video captioning model and each caption is added to context along with the frame number so temporal information can be maintained.
