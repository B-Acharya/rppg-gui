use nokhwa::Camera;
use nokhwa::pixel_format::*;
use nokhwa::utils::*;

fn main() {
    // first camera in system
    println!("Hello camera");
    // first camera in system
    let index = CameraIndex::Index(0);
    // request the absolute highest resolution CameraFormat that can be decoded to RGB.
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    // make the camera
    //
    let mut camera = match Camera::new(index, requested) {
        Ok(cam) => cam,
        Err(e) => {
            //TODO:need to check what eprintln is !
            eprintln!("Camera failed to initialize {e}");
            return;
        } //Also match is just to catch errors but without the return statement the above code doesnt
          //work as diffeerent types are returned ?
    };

    if let Err(e) = camera.open_stream() {
        eprintln!("Failed to open stream {e}");
        return;
    }

    // get a frame
    match camera.frame() {
        Ok(cam) => {
            println!("Captured Single Frame of {}", cam.buffer().len());
            let decoded = cam.decode_image::<RgbFormat>().unwrap();
            println!("Decoded Frame of {}", decoded.len());
        }
        Err(e) => {
            eprintln!("Failed to open stream {e}");
        }
    }

    // decode into an ImageBuffer
}

