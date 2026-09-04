"""Stage 1 entry point: start the object-oriented depth test.

Objects used by this program:
    DepthCamera          -> OpenNI2 connection and raw depth acquisition
    DepthFrameProcessor  -> depth processing and colour preview
    DepthTestApplication -> coordinates the complete test
"""

from applications import DepthTestApplication


if __name__ == "__main__":
    # Composition happens inside DepthTestApplication: the application object
    # creates and owns one camera object and one frame-processor object.
    application = DepthTestApplication(display_limit_mm=5000)

    # run() controls the program until Q/Escape is pressed.
    raise SystemExit(application.run())
