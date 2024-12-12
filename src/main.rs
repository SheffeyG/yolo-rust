use clap::Parser;
use image;
use std::sync::Arc;
use tokio::sync::Mutex;
use warp::{Filter, Rejection, Reply};
use yolo::{Args, YOLO};
use log::info;

#[derive(Debug)]
struct PredictionError {
    message: String,
}

impl warp::reject::Reject for PredictionError {}

async fn predict(
    model: Arc<Mutex<YOLO>>,
    image_path: String,
) -> Result<impl Reply, Rejection> {
    let im = image::io::Reader::open(&image_path)
        .map_err(|e| PredictionError {
            message: format!("Image open error: {}", e),
        })?
        .with_guessed_format()
        .map_err(|e| PredictionError {
            message: format!("Image format error: {}", e),
        })?
        .decode()
        .map_err(|e| PredictionError {
            message: format!("Image decode error: {}", e),
        })?;

    let imgs = vec![im];

    let mut model = model.lock().await;
    let res = model
        .run(&imgs)
        .map_err(|e| PredictionError {
            message: format!("Model prediction error: {}", e),
        })?;

    info!("Received image \"{image_path}\".");

    // Convert YOLOResult to a String representation
    let result_string = format!("{:?}", res);

    Ok(result_string)
}

async fn error_handler(err: Rejection) -> Result<impl Reply, Rejection> {
    if let Some(e) = err.find::<PredictionError>() {
        Ok(warp::reply::with_status(
            e.message.clone(),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ))
    } else {
        Ok(warp::reply::with_status(
            "Internal Server Error".to_string(),
            warp::http::StatusCode::INTERNAL_SERVER_ERROR,
        ))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();
    let model = Arc::new(Mutex::new(YOLO::new(args)?));

    info!("Detection server started!");

    let routes= warp::path!("predict" / String)
        .and(warp::get())
        .and(warp::any().map(move || Arc::clone(&model)))
        .and_then(|image_path, model| async { predict(model, image_path).await })
        .recover(error_handler);

    warp::serve(routes)
        .run(([127, 0, 0, 1], 3030))
        .await;

    Ok(())
}
