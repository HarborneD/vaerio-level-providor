import json

import numpy as np
import cv2
from numpy.lib.function_base import append 

import seaborn as sb
import matplotlib.pyplot as plt

from LevelSliceClient import GetLevelSlicesForVectors



class EnjoymentSurfaceContentGenerator():
    def __init__(self):
        self.name = "enjoyment-surface"
        self.generator_model_to_use = "mariovae_z_dim_2"

        self.max_player_memory = 20
        self.player_lookup = {}
        self.player_queue = []

        self.latent_dimension_bounds = np.array([
            (0, 0),
            (10, 10)
        ])

        self.surface_shape = np.array([300, 300])
        self.update_redius_ratio = np.array([0.05, 0.05])
        self.update_strength = 0.05


        self.latent_to_surface_multiplier = 1 / (self.latent_dimension_bounds[1] - self.latent_dimension_bounds[0])


        self.surface = np.full(self.surface_shape, 0.5)
        self.update_redius = np.rint(np.multiply(self.surface_shape, self.update_redius_ratio))
        self.update_shape = np.full(self.update_redius.astype(int).tolist(), self.update_strength)

        self.surface_cord_arrays = np.indices(self.surface_shape.astype(int).tolist())


    def UpdatePlayerRecords(self, player_request):
        if player_request["playerId"] in self.player_lookup:
            try:
                self.player_queue.remove(player_request["playerId"])
            except:
                pass
        else:
            self.player_lookup[player_request["playerId"]] = self.surface.copy()
            
        self.player_queue.append(player_request["playerId"])

        if(len(self.player_queue) > self.max_player_memory):
            removed_player = self.player_queue.pop(0)
            self.player_lookup.pop(removed_player, None)
        

        for latent_vector in player_request["telemetry"]["latentVectors"]:
            self.UpdateSurfaceWithLatentSpaceLocation(latent_vector, player_request["telemetry"]["surveyResults"]["enjoyment"])
            self.UpdateSurfaceWithLatentSpaceLocation(latent_vector, player_request["telemetry"]["surveyResults"]["enjoyment"], surfaceToUpdate=self.player_lookup[player_request["playerId"]])




    def DisplaySurface(self):
        cv2.imshow("Enjoyment Surface", self.surface)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    

    def DisplaySurfaceUsingSeaborn(self):
        heat_map_plot = sb.heatmap(self.surface, yticklabels=False, xticklabels=False)
        plt.show()
    

    def DrawSlices(self, draw_tuple):
        pos, w, max_w = draw_tuple
        wall_min = max(pos, 0)
        wall_max = min(pos+w, max_w)
        block_min = -min(pos, 0)
        block_max = max_w-max(pos+w, max_w)
        block_max = block_max if block_max != 0 else None
        return slice(wall_min, wall_max), slice(block_min, block_max)


    def AddShapeValuesToArrayLocation(self, target_array, shape, location):
        loc_zip = zip(location, shape.shape, target_array.shape)
        wall_slices, block_slices = zip(*map(self.DrawSlices, loc_zip))
        target_array[wall_slices] += shape[block_slices]


    def UpdateSurface(self, point, value, surfaceToUpdate=None):
        update_point = (point[0] - int(self.update_redius[0] / 2), point[1] - int(self.update_redius[1] / 2))
        if(surfaceToUpdate is None):
            self.AddShapeValuesToArrayLocation( self.surface, value * self.update_shape, update_point )
        else:
            self.AddShapeValuesToArrayLocation( surfaceToUpdate, value * self.update_shape, update_point )

    def MapLatentSpacePointToSurfacePoint(self, latent_space_point):
        surface_position_ratio = (latent_space_point - self.latent_dimension_bounds[0]) * self.latent_to_surface_multiplier
        return np.rint(np.multiply( self.surface_shape, surface_position_ratio)).astype(int)
    

    def MapSurfaceSpaceToLatentSpace(self, surface_space_point):
        surface_position_ratio = surface_space_point / self.surface_shape
        return self.latent_dimension_bounds[0] + (surface_position_ratio * (self.latent_dimension_bounds[1] - self.latent_dimension_bounds[0]))


    def UpdateSurfaceWithLatentSpaceLocation(self, latent_space_point, enjoyment_value, surfaceToUpdate=None):
        surface_point = self.MapLatentSpacePointToSurfacePoint(latent_space_point)
        self.UpdateSurface(surface_point, enjoyment_value, surfaceToUpdate=surfaceToUpdate)


    def GenerateNoveltyModifierMap(self, current_vector, desired_novelty, distance_amplifier = 5):
        #TODO: Do something smarter than modiying by uniform distance from current point (learn which directions in the space grant more novelty)
        surface_point = self.MapLatentSpacePointToSurfacePoint(current_vector)
        
        novelty_modifier = self.surface_cord_arrays

        #Calculate the euclidean distance between current point and all points in bounded latent space

        # calculate the difference on each dimension
        for dimension_i in range(len(current_vector)):
            novelty_modifier[dimension_i] -= surface_point[dimension_i]
        
        # complete the euclidean distance calculation 
        novelty_modifier = novelty_modifier**2
        novelty_modifier = np.sum(novelty_modifier, axis=0)
        novelty_modifier = np.sqrt(novelty_modifier)
        
        #normalise        
        novelty_modifier /= np.max(novelty_modifier)

        #change distribution to be 0-centered around distances from the current point == to the desired novelty
        novelty_modifier = np.ones(novelty_modifier.shape) - (distance_amplifier * np.abs(novelty_modifier - desired_novelty))

        return novelty_modifier

        
    def GenerateLevel(self, player_request, show_old_and_new=False):
        PROBABILITY_SCALING_FACTOR = 10
        CUTOFF_PERCENTAGE = 99
        NOVELTY_WEIGHTING = 0.7

        self.UpdatePlayerRecords(player_request)

        new_vectors_surface_space = []

        for latent_vector in player_request["telemetry"]["latentVectors"]:
            novelty_modifier_matrix = self.GenerateNoveltyModifierMap(latent_vector, player_request["telemetry"]["surveyResults"]["desiredNovelty"])
            
            new_vector_distribution = NOVELTY_WEIGHTING * novelty_modifier_matrix + (1-NOVELTY_WEIGHTING)*self.player_lookup[player_request["playerId"]] 

            # new_vector_distribution_plot = sb.heatmap(new_vector_distribution, yticklabels=False, xticklabels=False)
            # plt.show()

            new_vector_distribution = new_vector_distribution + abs(np.min(new_vector_distribution)) # make probabilities non-negative

            # new_vector_distribution_plot = sb.heatmap(new_vector_distribution, yticklabels=False, xticklabels=False)
            # plt.show()
            

            cutoff = np.percentile(new_vector_distribution, CUTOFF_PERCENTAGE)
            new_vector_distribution[new_vector_distribution<cutoff] = 0

            # new_vector_distribution_plot = sb.heatmap(new_vector_distribution, yticklabels=False, xticklabels=False)
            # plt.show()
            
            
            # new_vector_distribution**PROBABILITY_SCALING_FACTOR #exagerate differences

            # new_vector_distribution_plot = sb.heatmap(new_vector_distribution, yticklabels=False, xticklabels=False)
            # plt.show()
            
            new_vector_distribution /= new_vector_distribution.sum() #ensure probabilities add up to 1

            # plt.hist(new_vector_distribution)
            # plt.show()
            
            flat_index=np.random.choice( np.array(list(range(0, new_vector_distribution.shape[0] * new_vector_distribution.shape[1]))), 1, p=new_vector_distribution.flatten())

            new_vector = np.array(np.unravel_index(flat_index , new_vector_distribution.shape)).astype(int).tolist()
            new_vectors_surface_space.append([item for sublist in new_vector for item in sublist])
        
        if(show_old_and_new):
            old_vectors_surface_space =[self.MapLatentSpacePointToSurfacePoint(v).tolist() for v in player_request["telemetry"]["latentVectors"]]
            print(old_vectors_surface_space)
            print( new_vectors_surface_space )
        
            old_vectors_image = self.PlotVectorsOnSurface(old_vectors_surface_space, self.player_lookup[player_request["playerId"]])
            new_vectors_image = self.PlotVectorsOnSurface(new_vectors_surface_space, self.player_lookup[player_request["playerId"]])

            f, axarr = plt.subplots(1,2)
            axarr[0].imshow(old_vectors_image)
            axarr[1].imshow(new_vectors_image)
            plt.title("Novelty: " + str(player_request["telemetry"]["surveyResults"]["desiredNovelty"]))
            plt.show()
        
        new_vectors = [self.MapSurfaceSpaceToLatentSpace(surface_space_vector).tolist() for surface_space_vector in new_vectors_surface_space]
        print(new_vectors)
        
        level_representation = GetLevelSlicesForVectors(latent_vectors=new_vectors, experiment_name=self.name, generator_model_name=self.generator_model_to_use)
        
        return {
            "latentVectors":new_vectors,
            "levelRepresentation":level_representation
        }


    def PlotVectorsOnSurface(self, vectors, surfaceToUse):
        plot_image = surfaceToUse.copy()

        COLOR_LIST = [
            (213, 255, 0),
            # (255, 0, 86),
            # (158, 0, 142),
            # (14, 76, 161),
            # (255, 229, 2),
            # (0, 95, 57),
            # (0, 255, 0),
            # (149, 0, 58),
            # (255, 147, 126),
            (164, 36, 0),
            (0, 21, 68),
            (145, 208, 203),
            (98, 14, 0),
            (107, 104, 130),
            (0, 0, 255),
            (0, 125, 181),
            (106, 130, 108),
            (0, 174, 126),
            (194, 140, 159),
            (190, 153, 112),
            (0, 143, 156),
            (95, 173, 78),
            (255, 0, 0),
            (255, 0, 246),
            (255, 2, 157),
        ]

        for vector_i, vector in enumerate(vectors):
            plot_image = cv2.circle(plot_image, vector, 5, COLOR_LIST[vector_i], 2)
        
        return plot_image
            



def BuildSurfaceFromFeedbackData(generator):
    import requests


    FEEDBACK_MAX = 5

    half_feedback_max = FEEDBACK_MAX/2

    dataset_url = "https://vaerio-level-providor.herokuapp.com/feedback"

    resp = requests.get(dataset_url)
    feedbacks_json = resp.json()

    for feedback_json in feedbacks_json["feedbackItems"]:
        for latent_vector in feedback_json["latent_vectors"]:
            generator.UpdateSurfaceWithLatentSpaceLocation(latent_vector, feedback_json["enjoyment"]-half_feedback_max/half_feedback_max)

        
   



if __name__ == "__main__":
    import uuid

    #TODO: handle could not finish
    #TODO: handle did not finish
    #TODO: Generate Vectors 

    with open("test_data.json", "r") as f:
        test_level_data = json.loads(f.read())

    TEST_LEVEL_REQUEST = {
        "requestId":str(uuid.uuid4()),
        "playerId":"EndpointTest",
        "telemetry":{
            "latentVectors": test_level_data["latentVectors"],
            "levelRepresentation":test_level_data["levelRepresentation"],
            "modelName": "mariovae_z_dim_2",
            "experimentName": test_level_data["experimentName"],
            "markedUnplayable": False,
            "endedEarly": False,
            "surveyResults":{
                    "enjoyment": 0.5,
                    "ratedNovelty": 0.4,
                    "desiredNovelty": 0.2
                }
        }
    }

    
    generator = EnjoymentSurfaceContentGenerator()
    BuildSurfaceFromFeedbackData(generator)

    new_level = generator.GenerateLevel(TEST_LEVEL_REQUEST, show_old_and_new=True)
    new_level = generator.GenerateLevel(TEST_LEVEL_REQUEST, show_old_and_new=True)
    new_level = generator.GenerateLevel(TEST_LEVEL_REQUEST, show_old_and_new=True)
    new_level = generator.GenerateLevel(TEST_LEVEL_REQUEST, show_old_and_new=True)
    
    
    
    # generator.DisplaySurfaceUsingSeaborn()

    # generator.UpdateSurfaceWithLatentSpaceLocation(TEST_LEVEL_REQUEST["telemetry"]["latentVectors"][0], TEST_LEVEL_REQUEST["telemetry"]["surveyResults"]["enjoyment"])

    # generator.DisplaySurface()
