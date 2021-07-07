from typing import List
import psycopg2


class PlayerFeedbackTable:
    """
    The player feedback table
    """

    def __init__(self, db):
        """
        Takes a db connection.
        """
        self.table_name = "player_feedback"
        self.db = db


    def ExecuteQuery(self, query):
        c = self.db.cursor()
        c.execute(query)
        self.db.commit()

    def CreateTable(self):
        query = f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
        query += "id SERIAL PRIMARY KEY,"
        query += "timestamp FLOAT,"
        query += "player_id VARCHAR(100),"
        query += "experiment_name VARCHAR(100),"
        query += "model_name VARCHAR(100),"
        query += "latent_vectors FLOAT[][],"
        query += "level_representation FLOAT[][][],"
        query += "marked_unplayable BOOL,"
        query += "ended_early BOOL,"
        query += "enjoyment FLOAT,"
        query += "rated_novelty FLOAT,"
        query += "desired_novelty FLOAT"
        query += ")"
        try:
            self.ExecuteQuery(query)
        except psycopg2.errors.UniqueViolation:
            print("Table already exists?")
            pass


    def ParseLatentVectorsArray(self, latent_vecotrs):
        return "{" + ",".join(["{" + ",".join([ str(value) for value in latent_vector ]) + "}" for latent_vector in latent_vecotrs]) + "}"


    def ParseLevelRepresentationArray(self, level_representation):
        return   "{" + ",".join([  "{" + ",".join(["{" + ",".join([ str(value) for value in row ]) + "}" for row  in level_slice ]) + "}" for level_slice in level_representation]) + "}"


    def FormQueryString(self,
        timestamp: float,
        player_id: str,
        experiment_name: str,
        model_name: str,
        latent_vectors: List[List[List[float]]],
        latlevel_representation: List[List[List[List[float]]]],
        marked_unplayable: bool,
        ended_early: bool,
        enjoyment: float,
        rated_novelty: float,
        desired_novelty: float):

        query = f"INSERT INTO {self.table_name} (timestamp, player_id, experiment_name, model_name, latent_vectors, level_representation, marked_unplayable, ended_early, enjoyment, rated_novelty, desired_novelty) VALUES "
        query += f" ({timestamp}, '{player_id}', '{experiment_name}', '{model_name}', '{self.ParseLatentVectorsArray(latent_vectors)}', '{self.ParseLevelRepresentationArray(latlevel_representation)}', '{marked_unplayable}', '{ended_early}', '{enjoyment}', '{rated_novelty}', '{desired_novelty}');"
        return query
        

    def SaveFeedback(
        self,
        timestamp: float,
        player_id: str,
        experiment_name: str,
        model_name: str,
        latent_vectors: List[List[List[float]]],
        latlevel_representation: List[List[List[List[float]]]],
        marked_unplayable: bool,
        ended_early: bool,
        enjoyment: float,
        rated_novelty: float,
        desired_novelty: float
    ):
        query = self.FormQueryString(
        timestamp,
        player_id,
        experiment_name,
        model_name,
        latent_vectors,
        latlevel_representation,
        marked_unplayable,
        ended_early,
        enjoyment,
        rated_novelty,
        desired_novelty)
        
        self.ExecuteQuery(query)


    def GetAllItemsForExperiment(self, experiment_name: str):
        """
        Returns all feedback items for a given experiment name.
        """
        query = (
            f"SELECT * FROM {self.table_name} WHERE experiment_name='{experiment_name}'"
        )
        c = self.db.cursor()
        c.execute(query)
        queries = c.fetchall()
        return queries
    

    def GetAllItems(self):
        """
        Returns all feedback items.
        """
        query = (
            f"SELECT * FROM {self.table_name};"
        )
        c = self.db.cursor()
        c.execute(query)
        queries = c.fetchall()
        return queries
