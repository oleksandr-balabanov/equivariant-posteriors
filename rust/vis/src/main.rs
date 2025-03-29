use chrono::NaiveDateTime;
use clap::Parser;
use eframe::egui;
use egui::{epaint, Stroke};
use egui_plot::{Legend, PlotPoints};
use egui_plot::{Line, Plot};
use itertools::Itertools;
use sqlx::postgres::PgPoolOptions;
use sqlx::Row;
use std::borrow::Cow;
use tokio::task::JoinHandle;
// use sqlx::types::JsonValue
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::ops::RangeInclusive;
use std::sync::mpsc::{self, Receiver, Sender};

pub mod np;
use colorous::CIVIDIS;
use ndarray_stats::QuantileExt;
use np::load_npy;

#[derive(Parser, Debug)]
struct Args {
    #[arg(default_value = "../../")]
    artifacts: String,
}

#[derive(Default, Debug, Clone)]
struct Metric {
    xaxis: String,
    orig_values: Vec<[f64; 3]>,
    values: Vec<[f64; 2]>,
    resampled: Vec<[f64; 2]>,
}

#[derive(Default, Debug, Clone)]
struct Run {
    params: HashMap<String, String>,
    artifacts: HashMap<String, String>,
    metrics: HashMap<String, Metric>,
    created_at: chrono::NaiveDateTime,
}

#[derive(Default, Debug, Clone)]
struct Runs {
    runs: HashMap<String, Run>,
    active_runs: Vec<String>, // filtered_runs: HashMap<String, Run>,
    active_runs_time_ordered: Vec<(String, chrono::NaiveDateTime)>,
    time_filtered_runs: Vec<String>,
}

#[derive(Default, Debug, Clone)]
enum XAxis {
    #[default]
    Batch,
    Time,
}

#[derive(Default, Debug, Clone)]
struct GuiParams {
    n_average: usize,
    max_n: usize,
    param_filters: HashMap<String, HashSet<String>>,
    param_name_filter: String,
    metric_filters: HashSet<String>,
    artifact_filters: HashSet<String>,
    inspect_params: HashSet<String>,
    time_filter_idx: usize,
    time_filter: Option<chrono::NaiveDateTime>,
    x_axis: XAxis,
}

#[derive(PartialEq, Eq)]
enum DataStatus {
    Waiting,
    FirstDataArrived,
    FirstDataProcessed,
    FirstDataPlotted,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct ArtifactId {
    train_id: String,
    name: String,
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Clone)]
struct NPYArtifactView {
    artifact_id: ArtifactId,
    index: Vec<usize>,
}

enum NPYArray {
    Loading(JoinHandle<std::io::Result<ndarray::ArrayD<f32>>>),
    Loaded(ndarray::ArrayD<f32>),
    Error(String),
}

enum ArtifactHandler {
    NPYArtifact {
        textures: HashMap<NPYArtifactView, egui::TextureHandle>,
        arrays: HashMap<ArtifactId, NPYArray>,
        views: HashMap<ArtifactId, NPYArtifactView>,
    },
    ImageArtifact {
        images: HashMap<ArtifactId, String>,
    },
}

fn add_artifact(handler: &mut ArtifactHandler, run_id: &str, name: &str, path: &str, args: &Args) {
    let artifact_id = ArtifactId {
        train_id: run_id.to_string(),
        name: name.to_string(),
    };
    match handler {
        ArtifactHandler::NPYArtifact {
            arrays,
            textures: _,
            views: _,
        } => {
            // if arrays.contains_key(run_id) {
            // let args = Args::parse();
            let base_path = std::path::Path::new(&args.artifacts);
            let full_path = base_path.join(std::path::Path::new(path));
            let mpath = full_path.clone();
            // if !arrays.contains_key(&artifact_id) {
            match arrays.get(&artifact_id) {
                None => {
                    let shoop = tokio::spawn(async move {
                        let path = full_path.as_path();
                        load_npy(path)
                    });
                    arrays.insert(artifact_id, NPYArray::Loading(shoop));
                }
                Some(NPYArray::Loading(array_join_handle)) => {
                    if array_join_handle.is_finished() {
                        let Some(NPYArray::Loading(array_join_handle)) =
                            arrays.remove(&artifact_id)
                        else {
                            panic!();
                        };
                        let npyarray = tokio::runtime::Handle::current()
                            .block_on(array_join_handle)
                            .unwrap();
                        match npyarray {
                            Ok(new_array) => {
                                // let array = arrays.get_mut(run_id).unwrap();
                                // *array = new_array;
                                arrays.insert(artifact_id, NPYArray::Loaded(new_array));
                            }
                            Err(err) => {
                                arrays.insert(
                                    artifact_id,
                                    NPYArray::Error(
                                        format!("{} [{}]", err, mpath.display()).to_string(),
                                    ),
                                );
                            }
                        }
                    }
                }
                Some(NPYArray::Error(_err)) => {}
                Some(NPYArray::Loaded(_array)) => {}
            }
        }
        ArtifactHandler::ImageArtifact { images } => match images.get(&artifact_id) {
            Some(_) => {}
            None => {
                let base_path = std::path::Path::new(&args.artifacts);
                let full_path = base_path.join(std::path::Path::new(path));

                images.insert(artifact_id, full_path.to_string_lossy().to_string());
            }
        },
    }
}

fn image_from_ndarray_healpix(
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    view: &NPYArtifactView,
) -> egui::ColorImage {
    let max = array.max().unwrap();
    let min = array.min().unwrap();
    let t = |x: f32| (x - min) / (max - min);
    let width = 1000;
    let height = 1000;
    let mut img = egui::ColorImage::new([width, height], egui::Color32::WHITE);
    let mut local_index = vec![0; array.shape().len()];
    for (dim_idx, dim) in view.index.iter().enumerate() {
        local_index[dim_idx] = *dim;
    }
    let ndim = local_index.len();
    for y in 0..height {
        for x in 0..width {
            // let lon_x = x as f64 / width as f64 * 8.0;
            // let lat_y = (y as f64 - height as f64 / 2.0) / (height as f64 / 2.0) * 2.0;
            // let (lon, lat) = cdshealpix::unproj(lon_x, lat_y);
            let (lon, lat) = (
                x as f64 / width as f64 * 2.0 * std::f64::consts::PI,
                -(y as f64 - height as f64 / 2.0) / (height as f64 / 2.0) * std::f64::consts::PI
                    / 2.0,
            );
            let nside = ((array.shape().last().unwrap() / 12) as f32).sqrt() as u32;
            let depth = cdshealpix::depth(nside);
            let hp_idx = cdshealpix::nested::hash(depth, lon, lat);
            // if hp_idx < cdshealpix::n_hash(depth) {
            // println!("nside {}, depth {}, idx {}", nside, depth, hp_idx);
            // dbg!(array.shape());
            local_index[ndim - 1] = hp_idx as usize;
            let color =
                CIVIDIS.eval_continuous(t(*array.get(local_index.as_slice()).unwrap()) as f64);
            img.pixels[y * width + x] = egui::Color32::from_rgb(color.r, color.g, color.b);
            // }
        }
    }
    img
}

struct GuiRuns {
    runs: Runs,
    dirty: bool,
    db_train_runs_sender: Sender<Vec<String>>,
    db_reciever: Receiver<HashMap<String, Run>>,
    recomputed_reciever: Receiver<HashMap<String, Run>>,
    dirty_sender: Sender<(GuiParams, HashMap<String, Run>)>,
    initialized: bool,
    data_status: DataStatus,
    gui_params: GuiParams,
    artifact_handlers: HashMap<String, ArtifactHandler>,
    args: Args, // texture: Option<egui::TextureHandle>,
}

fn recompute(runs: &mut HashMap<String, Run>, gui_params: &GuiParams) {
    resample(runs, gui_params);
}

fn get_train_ids_from_filter(runs: &HashMap<String, Run>, gui_params: &GuiParams) -> Vec<String> {
    if gui_params
        .param_filters
        .values()
        .map(|hs| hs.is_empty())
        .all(|x| x)
    {
        return Vec::new();
    }
    runs.iter()
        .filter_map(|run| {
            for (param_name, values) in gui_params
                .param_filters
                .iter()
                .filter(|(_, vs)| !vs.is_empty())
            {
                if let Some(run_value) = run.1.params.get(param_name) {
                    if !values.contains(run_value) {
                        return None;
                    }
                } else {
                    return None;
                }
                // if let Some(time_filter) = gui_params.time_filter {
                //     if run.1.created_at < time_filter {
                //         return None;
                //     }
                // }
            }
            Some(run.0)
        })
        .cloned()
        .collect()
}

fn resample(runs: &mut HashMap<String, Run>, gui_params: &GuiParams) {
    for run in runs.values_mut() {
        for metric in run.metrics.values_mut() {
            let max_n = gui_params.max_n;
            metric.values = if metric.orig_values.len() > max_n {
                let window = metric.orig_values.len() as i64 / (2 * max_n) as i64;
                let window = window.min(1);
                let didx = metric.orig_values.len() as f64 / max_n as f64;
                (0..max_n)
                    .map(|idx| {
                        let mean_value = (-window..=window)
                            .map(|sub_idx| {
                                metric.orig_values
                                    [((didx * idx as f64) as i64 + sub_idx).max(0) as usize][1]
                            })
                            .sum::<f64>()
                            / (-window..=window).count() as f64;
                        // let middle_x =
                        // metric.orig_values[((didx * idx as f64) as i64).max(0) as usize][0];
                        let middle_x = match gui_params.x_axis {
                            XAxis::Batch => {
                                metric.orig_values[((didx * idx as f64) as i64).max(0) as usize][0]
                            }
                            XAxis::Time => {
                                metric.orig_values[((didx * idx as f64) as i64).max(0) as usize][2]
                            }
                        };

                        [middle_x, mean_value]
                    })
                    .collect()
            } else {
                match gui_params.x_axis {
                    XAxis::Batch => metric.orig_values.iter().map(|x| [x[0], x[1]]).collect(),
                    XAxis::Time => metric.orig_values.iter().map(|x| [x[2], x[1]]).collect(),
                }
            };
            let fvalues: Vec<[f64; 2]> = (0..metric.values.len())
                .map(|orig_idx| {
                    let window = if gui_params.n_average > 0 {
                        -(gui_params.n_average.min(orig_idx) as i32)
                            ..=(gui_params.n_average.min(metric.values.len() - 1 - orig_idx) as i32)
                    } else {
                        0..=0
                    };
                    // let vals: Vec<f64> = window
                    //     .map(|sub_idx| {
                    //         let idx = orig_idx as i32 + sub_idx;
                    //         metric.values[idx as usize][1]
                    //     })
                    //     .sorted_by(|a, b| a.partial_cmp(b).unwrap())
                    //     .collect();
                    // let mean_val = vals[vals.len() / 2];
                    let sum: f64 = window
                        .clone()
                        .map(|sub_idx| {
                            let idx = orig_idx as i32 + sub_idx;
                            metric.values[idx as usize][1]
                        })
                        .sum();
                    let mean_val = sum / window.count() as f64;
                    [
                        metric.values[orig_idx][0],
                        mean_val,
                        // metric.values[orig_idx][2],
                    ]
                })
                .collect();
            metric.resampled = fvalues;
        }
    }
}

impl eframe::App for GuiRuns {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.initialized {
            ctx.set_zoom_factor(1.0);
        }
        self.runs.active_runs = get_train_ids_from_filter(&self.runs.runs, &self.gui_params);
        self.runs.active_runs_time_ordered = self
            .runs
            .active_runs
            .iter()
            .cloned()
            .map(|train_id| {
                (
                    train_id.clone(),
                    self.runs.runs.get(&train_id).unwrap().created_at,
                )
            })
            .sorted_by_key(|(_train_id, created_at)| *created_at)
            .collect();
        self.runs.time_filtered_runs = self
            .runs
            .active_runs_time_ordered
            .iter()
            .cloned()
            .filter(|(_train_id, created_at)| {
                if let Some(time_filter) = self.gui_params.time_filter {
                    *created_at >= time_filter
                } else {
                    true
                }
            })
            .map(|(train_id, _)| train_id)
            .collect();
        if self.dirty {
            self.dirty_sender
                .send((self.gui_params.clone(), self.runs.runs.clone()))
                .expect("Failed to send dirty runs");
            self.db_train_runs_sender
                .send(self.runs.active_runs.clone())
                .expect("Failed to send train runs to db thread");
            self.dirty = false;
            // self.recompute();
        }
        if let Ok(new_runs) = self.recomputed_reciever.try_recv() {
            self.runs.runs = new_runs;
            if self.data_status == DataStatus::FirstDataArrived && self.runs.runs.len() > 0 {
                self.data_status = DataStatus::FirstDataProcessed;
            }
        }
        if let Ok(new_runs) = self.db_reciever.try_recv() {
            for train_id in new_runs.keys() {
                if !self.runs.runs.contains_key(train_id) {
                    let new_active = get_train_ids_from_filter(&new_runs, &self.gui_params);
                    self.db_train_runs_sender
                        .send(new_active)
                        .expect("Failed to send train runs to db thread");
                    break;
                }
            }
            self.dirty_sender
                .send((self.gui_params.clone(), new_runs))
                .expect("Failed to send dirty runs");
            if self.data_status == DataStatus::Waiting {
                self.data_status = DataStatus::FirstDataArrived;
            }
            // self.recompute();
        }

        let ensemble_colors: HashMap<String, egui::Color32> = self
            .runs
            .runs
            .values()
            .map(|run| label_from_active_inspect_params(run, &self.gui_params))
            .unique()
            .sorted()
            .enumerate()
            .map(|(idx, ensemble_id)| {
                let h = idx as f32 * 0.61;
                let color: egui::Color32 = epaint::Hsva::new(h, 0.85, 0.5, 1.0).into();
                (ensemble_id.clone(), color)
            })
            .collect();
        let run_ensemble_color: HashMap<String, egui::Color32> = self
            .runs
            .runs
            .values()
            .map(|run| {
                let train_id = run.params.get("train_id").unwrap();
                let ensemble_id = label_from_active_inspect_params(run, &self.gui_params); // run.params.get("ensemble_id").unwrap();
                (
                    train_id.clone(),
                    *ensemble_colors.get(&ensemble_id).unwrap(),
                )
            })
            .collect();
        let param_values = get_parameter_values(&self.runs, true);
        for param_name in param_values.keys() {
            if !self.gui_params.param_filters.contains_key(param_name) {
                self.gui_params
                    .param_filters
                    .insert(param_name.clone(), HashSet::new());
            }
        }
        // for run in &filtered_runs {
        //     println!("{:?}", run.1.params.get("epochs"))
        // }
        let filtered_values = get_parameter_values(&self.runs, false);
        let metric_names: Vec<String> = self
            .runs
            .time_filtered_runs
            .iter()
            .map(|train_id| self.runs.runs.get(train_id).unwrap())
            .map(|run| run.metrics.keys().cloned())
            .flatten()
            .unique()
            .sorted()
            .collect();

        egui::SidePanel::left("Controls")
            .resizable(true)
            .default_width(200.0)
            .min_width(200.0)
            // .width_range(100.0..=600.0)
            .show(ctx, |ui| {
                self.render_parameters(ui, param_values, filtered_values, ctx);
            });
        egui::SidePanel::right("Metrics")
            .resizable(true)
            .default_width(300.0)
            .width_range(100.0..=300.0)
            .show(ctx, |ui| {
                self.render_metrics(ui, &metric_names);
                ui.separator();
                self.render_artifact_selector(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_time_selector(ui);
            ui.separator();
            egui::ScrollArea::vertical()
                .id_source("central_space")
                .show(ui, |ui| {
                    self.render_artifacts(ui, &run_ensemble_color);
                    self.render_plots(ui, metric_names, run_ensemble_color);
                });
        });
        self.initialized = true;
        ctx.request_repaint();
    }
}

fn get_artifact_type(path: &String) -> &str {
    path.split(".").last().unwrap_or("unknown")
}

fn label_from_active_inspect_params(run: &Run, gui_params: &GuiParams) -> String {
    let label = if gui_params.inspect_params.is_empty() {
        run.params.get("ensemble_id").unwrap().clone()
    } else {
        let empty = "".to_string();
        gui_params
            .inspect_params
            .iter()
            .sorted()
            .map(|param| {
                format!(
                    "{}:{}",
                    param.split(".").last().unwrap_or(param),
                    run.params.get(param).unwrap_or(&empty)
                )
            })
            .join(", ")
    };
    label
}
fn show_artifacts(
    ui: &mut egui::Ui,
    handler: &mut ArtifactHandler,
    gui_params: &GuiParams,
    runs: &HashMap<String, Run>,
    filtered_runs: &Vec<String>,
    run_ensemble_color: &HashMap<String, egui::Color32>,
) {
    match handler {
        ArtifactHandler::NPYArtifact {
            arrays,
            textures,
            views,
        } => {
            // let texture = texture.get_or_insert_with(|| {});
            let npy_axis_id = ui.id().with("npy_axis");
            let available_artifact_names: Vec<&String> = arrays.keys().map(|id| &id.name).collect();
            for (artifact_name, filtered_arrays) in gui_params
                .artifact_filters
                .iter()
                .filter(|name| available_artifact_names.contains(name))
                .map(|name| {
                    (
                        name,
                        arrays.iter().filter(|(key, _v)| {
                            key.name == *name && filtered_runs.contains(&key.train_id)
                        }),
                    )
                })
            {
                // artifact_name,
                let plot_width = ui.available_width() * 0.48;
                let plot_height = ui.available_width() * 0.48 * 0.5;
                ui.horizontal_wrapped(|ui| {
                    for (artifact_name, array_group) in
                        &filtered_arrays.group_by(|(aid, _)| aid.name.clone())
                    {
                        ui.group(|ui| {
                            ui.label(egui::RichText::new(artifact_name).size(20.0));
                            ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                            for (artifact_id, array) in array_group {
                                match array {
                                    NPYArray::Loading(_) => {
                                        ui.label("loading...");
                                    }
                                    NPYArray::Loaded(array) => {
                                        // ui.allocate_ui()
                                        ui.allocate_ui(
                                            egui::Vec2::from([plot_width, plot_height + 200.0]),
                                            |ui| {
                                                render_npy_artifact(
                                                    ui,
                                                    runs,
                                                    artifact_id,
                                                    gui_params,
                                                    run_ensemble_color,
                                                    views,
                                                    array,
                                                    textures,
                                                    plot_width,
                                                    npy_axis_id,
                                                );
                                            },
                                        );
                                    }
                                    NPYArray::Error(err) => {
                                        // ui.colored_label(egui::Color32::RED, err);
                                        // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                                    }
                                }
                            }
                        });
                    }
                });
            }
        }
        ArtifactHandler::ImageArtifact { images } => {
            let max_size = egui::Vec2::new(ui.available_width() / 2.1, ui.available_height() / 2.1);
            let available_artifact_names: Vec<&String> = images.keys().map(|id| &id.name).collect();
            ui.horizontal_wrapped(|ui| {
                // ui.set_max_width(ui.available_width());
                // ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                for (artifact_name, filtered_arrays) in gui_params
                    .artifact_filters
                    .iter()
                    .filter(|name| available_artifact_names.contains(name))
                    .map(|name| {
                        (
                            name,
                            images.iter().filter(|(key, _v)| {
                                key.name == *name && filtered_runs.contains(&key.train_id)
                            }),
                        )
                    })
                {
                    for (artifact_id, uri) in filtered_arrays {
                        ui.push_id(uri, |ui| {
                            ui.vertical(|ui| {
                                let label = label_from_active_inspect_params(
                                    runs.get(&artifact_id.train_id).unwrap(),
                                    &gui_params,
                                );
                                ui.colored_label(
                                    run_ensemble_color
                                        .get(&artifact_id.train_id)
                                        .unwrap()
                                        .clone(),
                                    format!("{}: {}", artifact_id.name, label),
                                );
                                // ui.allocate_space(max_size);
                                ui.add(
                                    egui::Image::from_uri(format!("file://{}", uri))
                                        .fit_to_exact_size(max_size),
                                );
                            });
                        });
                    }
                }
            });
        }
    }
}

fn render_npy_artifact(
    ui: &mut egui::Ui,
    runs: &HashMap<String, Run>,
    artifact_id: &ArtifactId,
    gui_params: &GuiParams,
    run_ensemble_color: &HashMap<String, egui::Color32>,
    views: &mut HashMap<ArtifactId, NPYArtifactView>,
    array: &ndarray::prelude::ArrayBase<
        ndarray::OwnedRepr<f32>,
        ndarray::prelude::Dim<ndarray::IxDynImpl>,
    >,
    textures: &mut HashMap<NPYArtifactView, egui::TextureHandle>,
    plot_width: f32,
    npy_axis_id: egui::Id,
) {
    ui.vertical(|ui| {
        let label =
            label_from_active_inspect_params(runs.get(&artifact_id.train_id).unwrap(), &gui_params);
        ui.colored_label(
            run_ensemble_color
                .get(&artifact_id.train_id)
                .unwrap()
                .clone(),
            format!("{}: {}", artifact_id.name, label),
        );
        let view = views.entry(artifact_id.clone()).or_insert(NPYArtifactView {
            artifact_id: artifact_id.clone(),
            index: vec![0; array.shape().len() - 1],
        });
        for (dim_idx, dim) in array
            .shape()
            .iter()
            .enumerate()
            .take(array.shape().len() - 1)
        {
            ui.add(egui::Slider::new(&mut view.index[dim_idx], 0..=(dim - 1)));
        }
        ui.label(array.shape().iter().map(|x| x.to_string()).join(","));
        if !textures.contains_key(&view) {
            let mut texture = ui.ctx().load_texture(
                &artifact_id.name,
                egui::ColorImage::example(),
                egui::TextureOptions::default(),
            );
            let img = image_from_ndarray_healpix(array, view);
            texture.set(img, egui::TextureOptions::default());
            textures.insert(view.clone(), texture);
        }
        let pi = egui_plot::PlotImage::new(
            textures.get(&view).unwrap(),
            // texture.id(),
            egui_plot::PlotPoint::from([0.0, 0.0]),
            [2.0 * 3.14, 3.14],
        );
        // texture.set(img, egui::TextureOptions::default());
        // ui.image((texture.id(), texture.size_vec2()));
        Plot::new(artifact_id)
            .width(plot_width)
            .height(plot_width / 2.0)
            .data_aspect(1.0)
            .view_aspect(1.0)
            .show_grid(false)
            .link_axis(npy_axis_id, true, true)
            .link_cursor(npy_axis_id, true, true)
            .show(ui, |plot_ui| {
                plot_ui.image(pi);
            });
    });
}
impl GuiRuns {
    fn render_parameters(
        &mut self,
        ui: &mut egui::Ui,
        param_values: HashMap<String, HashSet<String>>,
        filtered_values: HashMap<String, HashSet<String>>,
        ctx: &egui::Context,
    ) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                if ui
                    .add(egui::TextEdit::singleline(
                        &mut self.gui_params.param_name_filter,
                    ))
                    .changed()
                {}
                if ui
                    .add(
                        egui::Slider::new(&mut self.gui_params.n_average, 0usize..=200usize)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    self.dirty = true;
                };
                if ui
                    .add(
                        egui::Slider::new(&mut self.gui_params.max_n, 500usize..=2000usize)
                            .logarithmic(true),
                    )
                    .changed()
                {
                    self.dirty = true;
                };
                ui.horizontal(|ui| {
                    if ui.button("Time").clicked() {
                        self.gui_params.x_axis = XAxis::Time;
                    }
                    if ui.button("Batch").clicked() {
                        self.gui_params.x_axis = XAxis::Batch;
                    }
                });
                let param_names = param_values.keys().cloned().collect_vec();
                self.render_parameters_one_level(
                    &param_values,
                    param_names,
                    ui,
                    &filtered_values,
                    ctx,
                    0,
                );
            });
        });
    }

    fn render_parameters_one_level(
        &mut self,
        param_values: &HashMap<String, HashSet<String>>,
        param_names: Vec<String>,
        ui: &mut egui::Ui,
        filtered_values: &HashMap<String, HashSet<String>>,
        ctx: &egui::Context,
        depth: usize,
    ) {
        let groups = param_names //param_values
            .iter()
            // .keys()
            .sorted()
            .group_by(|param_name| {
                param_name.split(".").take(depth + 1).join(".")
                // .unwrap_or(param_name)
            });
        for (param_group_name, param_group) in &groups {
            let param_group_vec = param_group
                .cloned()
                .map(|name| {
                    // if name.contains(".") {
                    // name.split_once(".").unwrap().1.to_string()
                    // name.replacen(".", "#", 1)
                    // } else {
                    name
                    // }
                })
                .collect_vec();
            if param_group_vec
                .iter()
                .all(|name| !name.contains(self.gui_params.param_name_filter.as_str()))
            {
                continue;
            }
            // ui.collapsing(param_group_name, |ui| {
            egui::CollapsingHeader::new(&param_group_name)
                // .default_open(self.gui_params.param_name_filter.len() > 1)
                .default_open(!param_group_name.ends_with("_id"))
                .show(ui, |ui| {
                    if param_group_vec
                        .iter()
                        .any(|name| name.split(".").count() > depth + 1)
                    {
                        // println!(
                        //     "{:?}: {}",
                        //     param_group_vec, self.gui_params.param_name_filter
                        // );
                        self.render_parameters_one_level(
                            param_values,
                            param_group_vec,
                            ui,
                            filtered_values,
                            ctx,
                            depth + 1,
                        );
                    } else {
                        for param_name in &param_group_vec {
                            if !param_name.contains(self.gui_params.param_name_filter.as_str()) {
                                continue;
                            }
                            // ui.
                            // ui.separator();
                            let frame_border =
                                if self.gui_params.inspect_params.contains(param_name) {
                                    1.0
                                } else {
                                    0.0
                                };
                            // ui.allocate_ui(egui::Vec2::new(0.0, 0.0), |ui| {})
                            let param_frame = egui::Frame::none()
                                // .fill(egui::Color32::GREEN)
                                .stroke(egui::Stroke::new(frame_border, egui::Color32::GREEN))
                                .show(ui, |ui| {
                                    ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                                    // ui.label(param_name);
                                    ui.horizontal_wrapped(|ui| {
                                        self.render_parameter_values(
                                            &param_values,
                                            &param_name,
                                            &filtered_values,
                                            ctx,
                                            ui,
                                        );
                                    });
                                });
                            // println!("{:?}", param_frame.response.sense);
                            if param_frame
                                .response
                                .interact(egui::Sense::click())
                                .clicked()
                            {
                                if self.gui_params.inspect_params.contains(param_name) {
                                    self.gui_params.inspect_params.remove(param_name);
                                } else {
                                    self.gui_params
                                        .inspect_params
                                        .insert(param_name.to_string());
                                }
                            }
                        }
                    }
                });
        }
    }

    fn render_parameter_values(
        &mut self,
        param_values: &HashMap<String, HashSet<String>>,
        param_name: &str,
        filtered_values: &HashMap<String, HashSet<String>>,
        ctx: &egui::Context,
        ui: &mut egui::Ui,
    ) {
        let param_name = param_name.replace("#", ".");
        for value in param_values
            .get(&param_name)
            .unwrap()
            .iter()
            .sorted_by(|name1, name2| {
                if let Ok(float1) = name1.parse::<f32>() {
                    if let Ok(float2) = name2.parse::<f32>() {
                        return float1.partial_cmp(&float2).unwrap();
                    }
                }
                name1.cmp(name2)
            })
        {
            let active_filter = self
                .gui_params
                .param_filters
                .get(&param_name)
                .unwrap()
                .contains(value);
            let filtered_runs_contains = if let Some(values) = filtered_values.get(&param_name) {
                values.contains(value)
            } else {
                false
            };
            let color = if filtered_runs_contains {
                egui::Color32::LIGHT_GREEN
            } else {
                ctx.style().visuals.widgets.inactive.bg_fill
            };
            if ui
                .add(
                    egui::Button::new(value)
                        .stroke(egui::Stroke::new(1.0, color))
                        .selected(active_filter),
                )
                .clicked()
            {
                if self
                    .gui_params
                    .param_filters
                    .get(&param_name)
                    .unwrap()
                    .contains(value)
                {
                    self.gui_params
                        .param_filters
                        .get_mut(&param_name)
                        .unwrap()
                        .remove(value);
                } else {
                    self.gui_params
                        .param_filters
                        .get_mut(&param_name)
                        .unwrap()
                        .insert(value.clone());
                    dbg!(&param_name, value);
                }
                self.dirty = true;
            }
        }
    }

    fn render_plots(
        &mut self,
        ui: &mut egui::Ui,
        metric_names: Vec<String>,
        run_ensemble_color: HashMap<String, egui::Color32>,
    ) {
        let xaxis_ids: HashMap<_, _> = self
            .runs
            .runs
            .values()
            .map(|run| run.metrics.values().map(|metric| metric.xaxis.clone()))
            .flatten()
            .unique()
            .sorted()
            .map(|xaxis| (xaxis.clone(), ui.id().with(xaxis)))
            .collect();

        let metric_name_axis_id: HashMap<_, _> = self
            .runs
            .runs
            .values()
            .map(|run| {
                run.metrics.iter().map(|(metric_name, metric)| {
                    (metric_name, xaxis_ids.get(&metric.xaxis).unwrap())
                })
            })
            .flatten()
            .unique()
            .collect();

        // let link_group_id = ui.id().with("linked_demo");
        let filtered_metric_names: Vec<String> = metric_names
            .into_iter()
            .filter(|name| {
                self.gui_params.metric_filters.contains(name)
                    || self.gui_params.metric_filters.is_empty()
            })
            .collect();
        let plot_width = if filtered_metric_names.len() <= 2 {
            ui.available_width() / 2.1
        } else {
            ui.available_width() / 2.1
        };
        let plot_height = if filtered_metric_names.len() <= 2 {
            ui.available_width() / 4.1
        } else {
            ui.available_width() / 4.1
        };
        let x_axis = self.gui_params.x_axis.clone();
        let plots: HashMap<_, _> = filtered_metric_names
            .into_iter()
            .map(|metric_name| {
                (
                    metric_name.clone(),
                    Plot::new(&metric_name)
                        .auto_bounds_x()
                        .auto_bounds_y()
                        .legend(Legend::default())
                        .width(plot_width)
                        .height(plot_height)
                        .x_axis_formatter(match x_axis {
                            XAxis::Time => |value, n_chars, range: &RangeInclusive<f64>| {
                                let ts =
                                    NaiveDateTime::from_timestamp_opt(value as i64, 0).unwrap();
                                let delta = range.end() - range.start();
                                if delta > (5 * 24 * 60 * 60) as f64 {
                                    ts.format("%m/%d").to_string()
                                } else if delta > (5 * 60 * 60) as f64 {
                                    ts.format("%d-%Hh").to_string()
                                } else {
                                    ts.format("%Hh:%Mm").to_string()
                                }
                            },
                            XAxis::Batch => |value, n_chars, range: &RangeInclusive<f64>| {
                                format!("{}", value as i64).to_string()
                            },
                        })
                        .link_axis(
                            **metric_name_axis_id.get(&metric_name).unwrap(),
                            true,
                            false,
                        )
                        .link_cursor(**metric_name_axis_id.get(&metric_name).unwrap(), true, true),
                )
            })
            .collect();
        // egui::ScrollArea::vertical().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            for (metric_name, plot) in plots.into_iter().sorted_by_key(|(k, _v)| k.clone()) {
                ui.allocate_ui(egui::Vec2::from([plot_width, plot_height]), |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(&metric_name);
                        plot.show(ui, |plot_ui| {
                            if self.gui_params.n_average > 1 {
                                for (run_id, run) in self
                                    .runs
                                    .time_filtered_runs
                                    .iter()
                                    .sorted()
                                    .map(|train_id| {
                                        (train_id, self.runs.runs.get(train_id).unwrap())
                                    })
                                {
                                    if let Some(metric) = run.metrics.get(&metric_name) {
                                        // let label = self.label_from_active_inspect_params(run);
                                        plot_ui.line(
                                            Line::new(PlotPoints::from(metric.values.clone()))
                                                // .name(&label)
                                                .stroke(Stroke::new(
                                                    1.0,
                                                    run_ensemble_color
                                                        .get(run_id)
                                                        .unwrap()
                                                        .gamma_multiply(0.4),
                                                )),
                                        );
                                    }
                                }
                            }
                            for (run_id, run) in self
                                .runs
                                .time_filtered_runs
                                .iter()
                                .sorted()
                                .map(|train_id| (train_id, self.runs.runs.get(train_id).unwrap()))
                            {
                                if let Some(metric) = run.metrics.get(&metric_name) {
                                    let label =
                                        label_from_active_inspect_params(run, &self.gui_params);
                                    plot_ui.line(
                                        Line::new(PlotPoints::from(metric.resampled.clone()))
                                            .name(&label)
                                            .stroke(Stroke::new(
                                                2.0,
                                                *run_ensemble_color.get(run_id).unwrap(),
                                            )),
                                    );
                                }
                            }
                        })
                    });
                    // });
                });
            }
        });
        // });
        if self.data_status == DataStatus::FirstDataProcessed {
            // plot_ui.set_auto_bounds(egui::Vec2b::new(true, true));
            self.data_status = DataStatus::FirstDataPlotted;
        }
    }

    fn render_metrics(&mut self, ui: &mut egui::Ui, metric_names: &Vec<String>) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.vertical(|ui| {
                for metric_name in metric_names {
                    let active_filter = self.gui_params.metric_filters.contains(metric_name);
                    if ui
                        .add(egui::Button::new(metric_name).selected(active_filter))
                        .clicked()
                    {
                        if self.gui_params.metric_filters.contains(metric_name) {
                            self.gui_params.metric_filters.remove(metric_name);
                        } else {
                            self.gui_params.metric_filters.insert(metric_name.clone());
                        }
                        self.dirty = true;
                    }
                }
            });
        });
    }

    fn render_artifact_selector(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_source("artifact_selector")
            .show(ui, |ui| {
                ui.allocate_space(egui::Vec2::new(ui.available_width(), 0.0));
                for artifact_name in self
                    .runs
                    .time_filtered_runs
                    // .active_runs
                    .iter()
                    .map(|train_id| self.runs.runs.get(train_id).unwrap().artifacts.keys())
                    .flatten()
                    .unique()
                    .sorted()
                {
                    if ui
                        .add(
                            egui::Button::new(artifact_name)
                                .selected(self.gui_params.artifact_filters.contains(artifact_name)),
                        )
                        .clicked()
                    {
                        if self.gui_params.artifact_filters.contains(artifact_name) {
                            self.gui_params.artifact_filters.remove(artifact_name);
                        } else {
                            self.gui_params
                                .artifact_filters
                                .insert(artifact_name.clone());
                        }
                    }
                }
            });
    }

    fn render_artifacts(
        &mut self,
        ui: &mut egui::Ui,
        run_ensemble_color: &HashMap<String, egui::Color32>,
    ) {
        let active_artifact_types: Vec<&str> = self
            .gui_params
            .artifact_filters
            .iter()
            .map(|artifact_name| {
                self.runs
                    .time_filtered_runs
                    .iter()
                    .map(|train_id| (train_id, self.runs.runs.get(train_id).unwrap()))
                    .map(|(_train_id, run)| {
                        if let Some(path) = run.artifacts.get(artifact_name) {
                            let artifact_type_str = get_artifact_type(path);
                            if self.artifact_handlers.contains_key(artifact_type_str) {
                                // println!("{}", artifact_type_str);
                                // add_artifact(handler, ui, train_id, path);
                                artifact_type_str
                            } else {
                                ""
                            }
                            // if let Some(artifact_handler) = self.
                        } else {
                            ""
                        }
                    })
            })
            .flatten()
            .unique()
            .sorted()
            .collect();
        for artifact_type in active_artifact_types {
            if let Some(handler) = self.artifact_handlers.get_mut(artifact_type) {
                for (run_id, run) in self
                    .runs
                    .time_filtered_runs
                    .iter()
                    .map(|run_id| (run_id, self.runs.runs.get(run_id).unwrap()))
                {
                    for (artifact_name, path) in run.artifacts.iter() {
                        if artifact_type == get_artifact_type(path) {
                            // println!("{}", artifact_type);
                            add_artifact(handler, run_id, artifact_name, path, &self.args);
                        }
                    }
                }
                show_artifacts(
                    ui,
                    handler,
                    &self.gui_params,
                    &self.runs.runs,
                    &self.runs.time_filtered_runs,
                    &run_ensemble_color,
                );
            }
            // if let Some(handler) = self.artifact_handlers.get(artifact_type) {}
        }
    }

    fn render_time_selector(&mut self, ui: &mut egui::Ui) {
        if self.runs.active_runs.len() > 1 {
            ui.group(|ui| {
                ui.spacing_mut().slider_width = ui.available_width() - 300.0;
                ui.label("Cut-off time");
                let time_slider = egui::Slider::new(
                    &mut self.gui_params.time_filter_idx,
                    0..=self.runs.active_runs.len() - 1,
                )
                .custom_formatter(|fval, _| {
                    let idx = fval as usize;
                    let created_at = self.runs.active_runs_time_ordered[idx].1;
                    created_at.to_string()
                });
                if ui.add(time_slider).changed() {
                    self.gui_params.time_filter =
                        Some(self.runs.active_runs_time_ordered[self.gui_params.time_filter_idx].1)
                }
            });
        }
    }
}

fn get_parameter_values(
    runs: &Runs, // active_runs: &Vec<String>,
    all: bool,
) -> HashMap<String, HashSet<String>> {
    let train_ids: Vec<String> = if all {
        runs.runs.keys().cloned().collect()
    } else {
        runs.active_runs.clone()
    };
    let mut param_values: HashMap<String, HashSet<String>> = train_ids
        .iter()
        .map(|train_id| runs.runs.get(train_id).unwrap())
        .map(|run| run.params.keys().cloned())
        .flatten()
        .unique()
        .map(|param_name| (param_name, HashSet::new()))
        .collect();
    for run in train_ids
        .iter()
        .map(|train_id| runs.runs.get(train_id).unwrap())
    {
        for (k, v) in &run.params {
            let values = param_values.get_mut(k).unwrap();
            values.insert(v.clone());
        }
    }
    param_values
}

async fn get_state_new(
    pool: &sqlx::postgres::PgPool,
    runs: &mut HashMap<String, Run>,
    train_ids: &Vec<String>,
    last_timestamp: &mut NaiveDateTime,
) -> Result<(), sqlx::Error> {
    // Query to get the table structure
    // TODO: WHERE train_id not in runs.keys()
    get_state_parameters(pool, runs).await?;
    get_state_artifacts(train_ids, pool, runs).await?;
    get_state_metrics(train_ids, last_timestamp, pool, runs).await?;
    Ok(())
}

async fn get_state_metrics(
    train_ids: &Vec<String>,
    last_timestamp: &mut NaiveDateTime,
    pool: &sqlx::Pool<sqlx::Postgres>,
    runs: &mut HashMap<String, Run>,
) -> Result<(), sqlx::Error> {
    Ok(if train_ids.len() > 0 {
        let mut batch_timestamp = last_timestamp.clone();
        let q = format!(
            r#"
        SELECT * FROM metrics WHERE train_id = ANY($1) AND created_at > $2 AND xaxis='batch' AND (CAST(x as int) % 1000 = 0 OR x < 500) ORDER BY train_id, variable, xaxis, x
        "#,
        );
        let metric_rows = sqlx::query(q.as_str())
            .bind(train_ids)
            .bind(batch_timestamp)
            .fetch_all(pool)
            .await?;
        parse_metric_rows(metric_rows, runs, &mut batch_timestamp);

        let mut epoch_timestamp = last_timestamp.clone();
        let q = format!(
            r#"
        SELECT * FROM metrics WHERE train_id = ANY($1) AND created_at > $2 AND xaxis='epoch' ORDER BY train_id, variable, xaxis, x
        "#,
        );
        let metric_rows = sqlx::query(q.as_str())
            .bind(train_ids)
            .bind(epoch_timestamp)
            .fetch_all(pool)
            .await?;
        parse_metric_rows(metric_rows, runs, &mut epoch_timestamp);
        *last_timestamp = batch_timestamp.max(epoch_timestamp);
    })
}

async fn get_state_artifacts(
    train_ids: &Vec<String>,
    pool: &sqlx::Pool<sqlx::Postgres>,
    runs: &mut HashMap<String, Run>,
) -> Result<(), sqlx::Error> {
    let artifact_rows = sqlx::query(
        r#"
        SELECT * FROM artifacts WHERE train_id = ANY($1) ORDER BY train_id
        "#,
    )
    .bind(train_ids)
    .fetch_all(pool)
    .await?;
    Ok(
        for (train_id, rows) in &artifact_rows
            .into_iter()
            .group_by(|row| row.get::<String, _>("train_id"))
        {
            for row in rows {
                let incoming_name: String = row.try_get("name").unwrap_or_default();
                let incoming_path: String = row.try_get("path").unwrap_or_default();
                if let Some(run) = runs.get_mut(&train_id) {
                    if let Some(path) = run.artifacts.get_mut(&incoming_name) {
                        *path = incoming_path;
                    } else {
                        run.artifacts.insert(incoming_name, incoming_path);
                    }
                } else {
                    println!("[Artifact] No run_id {}", train_id);
                }
            }
        },
    )
}

async fn get_state_parameters(
    pool: &sqlx::Pool<sqlx::Postgres>,
    runs: &mut HashMap<String, Run>,
) -> Result<(), sqlx::Error> {
    let run_rows = sqlx::query(
        r#"
        SELECT * FROM runs ORDER BY train_id
        "#,
    )
    .fetch_all(pool)
    .await?;
    Ok(
        for (train_id, db_params) in &run_rows
            .into_iter()
            .group_by(|row| row.get::<String, _>("train_id"))
        {
            let mut created_at: Option<chrono::NaiveDateTime> = None;

            let params: HashMap<_, _> = db_params
                .map(|row| {
                    if created_at.is_none() {
                        created_at = row.get("created_at");
                    }
                    (
                        row.get::<String, _>("variable"),
                        row.get::<String, _>("value_text"),
                    )
                })
                .collect();
            if !runs.contains_key(&train_id) {
                runs.insert(
                    train_id,
                    Run {
                        params,
                        metrics: HashMap::new(),
                        artifacts: HashMap::new(),
                        created_at: created_at.expect("No datetime for run parameters"),
                    },
                );
            }
        },
    )
}

fn parse_metric_rows(
    metric_rows: Vec<sqlx::postgres::PgRow>,
    runs: &mut HashMap<String, Run>,
    last_timestamp: &mut NaiveDateTime,
) {
    for (train_id, run_metric_rows) in &metric_rows
        .into_iter()
        .group_by(|row| row.get::<String, _>("train_id"))
    {
        let run = runs.get_mut(&train_id).unwrap();
        for (variable, value_rows) in
            &run_metric_rows.group_by(|row| row.get::<String, _>("variable"))
        {
            let rows: Vec<_> = value_rows.collect();
            let max_timestamp = rows
                .iter()
                .max_by_key(|row| row.get::<NaiveDateTime, _>("created_at"))
                .map(|row| row.get::<NaiveDateTime, _>("created_at"));
            if let Some(max_timestamp) = max_timestamp {
                if max_timestamp > *last_timestamp {
                    *last_timestamp = max_timestamp
                }
            }
            // println!("{:?}", rows[0].columns());
            let orig_values: Vec<_> = rows
                .iter()
                .filter_map(|row| {
                    // println!("{:?}", row.try_get::<i32, _>("id"));
                    if let Ok(x) = row.try_get::<f64, _>("x") {
                        if let Ok(value) = row.try_get::<f64, _>("value") {
                            if let Ok(created_at) = row.try_get::<NaiveDateTime, _>("created_at") {
                                return Some([x, value, created_at.timestamp() as f64]);
                            }
                        }
                    }
                    None
                })
                .collect();
            let xaxis = rows[0].get::<String, _>("xaxis");
            if !run.metrics.contains_key(&variable) {
                run.metrics.insert(
                    variable,
                    Metric {
                        resampled: Vec::new(),
                        orig_values,
                        xaxis,
                        values: Vec::new(),
                    },
                );
            } else {
                run.metrics
                    .get_mut(&variable)
                    .unwrap()
                    .orig_values
                    .extend(orig_values);
            }
        }
    }
}
// #[tokio::main(flavor = "current_thread")]
fn main() -> Result<(), sqlx::Error> {
    // Load environment variables
    // dotenv::dotenv().ok();
    let args = Args::parse();
    println!("Args: {:?}", args);
    let (tx, rx) = mpsc::channel();
    let (tx_gui_dirty, rx_gui_dirty) = mpsc::channel();
    let (tx_gui_recomputed, rx_gui_recomputed) = mpsc::channel();
    let (tx_db_filters, rx_db_filters) = mpsc::channel();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let rt_handle = rt.handle().clone();
    let _guard = rt.enter();

    std::thread::spawn(move || loop {
        if let Ok((gui_params, mut runs)) = rx_gui_dirty.recv() {
            recompute(&mut runs, &gui_params);
            tx_gui_recomputed
                .send(runs)
                .expect("Failed to send recomputed runs.");
        }
    });
    // Connect to the database
    // std::thread::spawn(move || {
    rt_handle.spawn(async move {
        let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&database_url)
            .await
            .expect("Can't connect to database");
        let mut train_ids = Vec::new();
        let mut last_timestamp = NaiveDateTime::from_timestamp_millis(0).unwrap();
        let mut runs: HashMap<String, Run> = Default::default();
        loop {
            // Fetch data from database
            println!("Getting state...");
            // get_state_new(&pool, &mut runs, &train_ids, &mut last_timestamp)
            //     .await
            //     .expect("Get state:");
            get_state_parameters(&pool, &mut runs).await;
            tx.send(runs.clone()).expect("Failed to send data");
            get_state_artifacts(&train_ids, &pool, &mut runs).await;
            tx.send(runs.clone()).expect("Failed to send data");
            get_state_metrics(&train_ids, &mut last_timestamp, &pool, &mut runs).await;

            // println!("{:?}", runs);
            println!("Done.");
            // Send data to UI thread
            // if let Ok(runs) = runs e
            tx.send(runs.clone()).expect("Failed to send data");
            // }
            // Wait some time before fetching again
            for _ in 0..100 {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                if let Ok(new_train_ids) = rx_db_filters.try_recv() {
                    train_ids = new_train_ids;
                    last_timestamp = NaiveDateTime::from_timestamp_millis(0).unwrap();
                    runs = HashMap::new();
                    break;
                }
            }
        }
    });
    // });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([800.0, 800.0]),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "Visualizer",
        options,
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::<GuiRuns>::new(GuiRuns {
                runs: Default::default(),
                dirty: true,
                db_train_runs_sender: tx_db_filters,
                db_reciever: rx,
                recomputed_reciever: rx_gui_recomputed,
                dirty_sender: tx_gui_dirty,
                initialized: false,
                data_status: DataStatus::Waiting,
                gui_params: GuiParams {
                    max_n: 1000,
                    param_filters: HashMap::new(),
                    metric_filters: HashSet::new(),
                    inspect_params: HashSet::new(),
                    n_average: 1,
                    artifact_filters: HashSet::new(),
                    time_filter: None,
                    time_filter_idx: 0,
                    x_axis: XAxis::Batch,
                    param_name_filter: "".to_string(),
                },
                // texture: None,
                artifact_handlers: HashMap::from([
                    (
                        "npy".to_string(),
                        ArtifactHandler::NPYArtifact {
                            arrays: HashMap::new(),
                            textures: HashMap::new(),
                            views: HashMap::new(),
                        },
                    ),
                    (
                        "png".to_string(),
                        ArtifactHandler::ImageArtifact {
                            images: HashMap::new(),
                        },
                    ),
                ]),
                args,
            })
        }),
    );

    Ok(())
}
