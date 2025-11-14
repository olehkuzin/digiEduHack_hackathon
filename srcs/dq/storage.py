from prompt import map_feature

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
from embedding import Embedding
import uuid

class Storage:
    def __init__(self, name, embedding_size):  
        """
        Initializes the Storage class

        Connects to the Qdrant client and ensures the collection exists,
        creating it with the specified configuration if it doesn't

        Args:
            name (str): The name of the collection to manage
            embedding_size (int): The dimensionality of the vectors (e.g., 384)
        """
        self.client = QdrantClient(host="localhost", port=6333, timeout=60.0)
        self.collection_name = name

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )
    
    def load_vectors_in_batches(self, vectors, texts, batch_size=100):
        """
        Loads large sets of vectors and texts in smaller batches

        This is a convenience wrapper around `load_vectors` to prevent
        sending payloads that are too large

        Args:
            vectors (list[np.ndarray]): A list of numpy array vectors
            texts (list[str]): A list of corresponding text payloads
            batch_size (int, optional): The number of points to upload per batch
                                    Defaults to 100
        """
        for i in range(0, len(vectors), batch_size):
            data = texts[i : i + batch_size]
            embedded_data = vectors[i : i + batch_size]

            self.load_vectors(embedded_data, data)

    def load_vectors(self, vectors, texts):
        """
        Upserts a batch of vectors and text payloads into the collection

        Generates a deterministic UUIDv5 for each point based on its text
        content to ensure uniqueness

        Args:
            vectors (list[np.ndarray]): A list of numpy array vectors
            texts (list[str]): A list of corresponding text payloads
        """
        namespace = uuid.NAMESPACE_DNS 
        
        self.client.upsert(
            collection_name = self.collection_name,
            points=[
                PointStruct(
                        id=str(uuid.uuid5(namespace, texts[idx])), 
                        vector=vector.tolist(),
                        payload={"col": texts[idx]}
                )
                for idx, vector in enumerate(vectors)
            ],
            wait=True
        )

    def smart_load(self, vector, text, feature_values, threshold=0.8):
        """
        Intelligently loads a single vector, checking for duplicates first

        1. Searches for the most similar existing vector
        2. If a similar vector (score > threshold) is found, it returns (False, existing_name)
        3. If unique, it consults an LLM (map_feature) to map the new text
        4. If the LLM identifies it as new ("NAN"), the vector is loaded and
           it returns (True, new_name)
        5. If the LLM maps it to an existing feature, it returns (False, mapped_name)

        Args:
            vector (np.ndarray): The single vector to check
            text (str): The corresponding text (e.g., column name)
            feature_values (list): A list of sample values from the column
            threshold (float, optional): The cosine similarity threshold for
                                         considering a vector a duplicate.
                                         Defaults to 0.8.

        Returns:
            tuple[bool, str]: A tuple of (was_added, name_to_use)
                              - (True, text) if the vector was new and added
                              - (False, existing_name) if a duplicate was found
                              - (False, mapped_name) if the LLM mapped it
        """
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            limit=1
        )

        if hits and hits[0].score > threshold:
            existing_name = hits[0].payload.get("col", "Unknown")
            return False, existing_name
        else:
            llm_res = map_feature(text, feature_values, self.list_data())
            if llm_res == "NAN":
                self.load_vectors_in_batches([vector], [text])
                return True, text
            
            return False, llm_res

    def search_similarities(self, query_vector, limit):
        """
        Searches the collection for vectors similar to the query vector

        Args:
            query_vector (np.ndarray): The vector to search against
            limit (int): The maximum number of similar points to return

        Returns:
            list[ScoredPoint]: A list of search results (hits)
        """
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        return hits
    
    def my_size(self):
        """
        Gets the exact total count of points in the collection

        Returns:
            int: The total number of points
        """
        return self.client.count(
            collection_name=self.collection_name, 
            exact=True
        )

    def my_info(self,):
        """
        Retrieves collection-level information and configuration

        Returns:
            CollectionInfo: An object containing collection details
        """
        return self.client.get_collection(
            collection_name=self.collection_name
        )
    
    def get_all_vectors(self, page_size=100):
        """
        Retrieves all points from the collection using pagination (scroll)

        This is used to get the entire dataset, not just search results

        Args:
            page_size (int, optional): How many points to retrieve per request
                                     Defaults to 100

        Returns:
            list[Record]: A list of all points in the collection
        """
        all_points = []
        offset = None

        while True:
            points, next_page_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=page_size,
                with_payload=True,
                with_vectors=True,
                offset=offset
            )
            
            all_points.extend(points)
            
            if next_page_offset is None:
                break
            
            offset = next_page_offset
            
        return all_points
    
    def list_data(self):
        """
        Gets a simple list of all text payloads stored in the collection

        This is a helper function that calls `get_all_vectors` and extracts
        the "col" payload from each point

        Returns:
            list[str]: A list of all stored "col" text values
        """
        ret = []
        all_stored_data = self.get_all_vectors()

        for point in all_stored_data:
            ret.append(point.payload.get("col"))

        return ret

