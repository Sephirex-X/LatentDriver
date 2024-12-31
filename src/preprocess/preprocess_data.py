# Initialize the environment
import src.utils.init_default_jax
import hydra
import os
from omegaconf import OmegaConf
from src.utils.utils import update_waymax_config
from waymax import dataloader
import numpy as np
from waymax.config import DatasetConfig
from src.preprocess.waymax_preprocess_utils import workers, get_whole_map, get_route_global
import multiprocessing as mp
import time

# Constants
MAP_DIR_NAME = 'map'
ROUTE_DIR_NAME = 'route'

class Preprocessor(object):
    def __init__(self, config):
        """
        Initialize the Preprocessor with the given configuration.
        
        Args:
            config (OmegaConf): The configuration object.
        """
        # Ensure that we are processing the whole dataset and not a customized subset
        waymax_conf = config.waymax_conf
        assert waymax_conf.pop('customized') == False, "You can only use this script to process the whole dataset."
        self.data_conf = config.data_conf
        
        # Initialize data iterator
        self.data_iter = dataloader.simulator_state_generator(config=DatasetConfig(**waymax_conf))
        
        # Set paths
        self.path_to_map = os.path.join(self.data_conf.path_to_processed_map_route, MAP_DIR_NAME)
        self.path_to_route = os.path.join(self.data_conf.path_to_processed_map_route, ROUTE_DIR_NAME)
        self.intention_label_path = config.metric_conf.intention_label_path

    def _check_and_create_dirs(self):
        """
        Check if directories exist and create them if not.
        
        Raises:
            ValueError: If any of the output directories already exist.
        """
        if os.path.exists(self.path_to_map):
            raise ValueError(f'The map has been dumped in {self.path_to_map}, please delete the map first')
        if os.path.exists(self.path_to_route):
            raise ValueError(f'The route has been dumped in {self.path_to_route}, please delete the route first')
        if os.path.exists(self.intention_label_path):
            raise ValueError(f'The intention label has been dumped in {self.intention_label_path}, please delete the intention label first')
        
        os.makedirs(self.path_to_map, exist_ok=True)
        os.makedirs(self.path_to_route, exist_ok=True)
        os.makedirs(self.intention_label_path, exist_ok=True)

    def _process_scenario(self, scen):
        """
        Process a single scenario to extract map, route, and intention label data.
        
        Args:
            scen: A scenario object.
            
        Returns:
            List of tuples containing data for each batch to be processed by workers.
        """
        cur_id = scen._scenario_id.reshape(-1)
        
        # Extract map data
        road_obs, ids = get_whole_map(scen)
        
        # Extract route data
        routes, ego_car_width = get_route_global(scen)
        routes = np.array(routes)
        ego_car_width = float(ego_car_width)
        
        # Extract intention label data
        mask = scen.object_metadata.is_sdc
        sdc_xy = np.array(scen.log_trajectory.xy[mask, ...])
        yaw = np.array(scen.log_trajectory.yaw[mask, ...])
        
        tasks = []
        for bs in range(len(cur_id)):
            tasks.append((
                # Map data
                road_obs[bs],
                ids[bs],
                self.data_conf.max_map_segments,
                os.path.join(self.path_to_map, '{}'.format(cur_id[bs])),
                # Route data
                routes[bs:bs+1],
                self.data_conf.max_route_segments,
                ego_car_width,
                os.path.join(self.path_to_route, '{}'.format(cur_id[bs])),
                # Intention label data
                sdc_xy[bs],
                yaw[bs],
                cur_id[bs],
                self.intention_label_path
            ))
        
        return tasks

    def run(self):
        """
        Run the preprocessing pipeline.
        """
        self._check_and_create_dirs()
        
        print(f'Start dumping whole map, the map will be saved in {self.path_to_map}')
        print(f'Start dumping route, the route will be saved in {self.path_to_route}')
        print(f'Start dumping intention label, the intention label will be saved in {self.intention_label_path}')
        
        with mp.Pool(processes=mp.cpu_count()) as pool:
            for batch_id, scen in enumerate(self.data_iter):
                t_start = time.time()
                tasks = self._process_scenario(scen)
                pool.starmap(workers, tasks)
                
                print(f"Processed; current batch is: {batch_id}; Using time is: {time.time() - t_start}")

@hydra.main(version_base=None, config_path="../../configs", config_name="simulate")
def run(cfg):
    """
    Entry point for the preprocessing script.
    
    Args:
        cfg (OmegaConf): The configuration object.
    """
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = update_waymax_config(cfg)
    preprocessor = Preprocessor(cfg)
    preprocessor.run()

if __name__ == '__main__':
    run()