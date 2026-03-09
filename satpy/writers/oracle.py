"""Module for writing feature datasets to Oracle databases."""
import oracledb
import shapely

from satpy.writers import Writer


class oracle(Writer):
    """Write features in geodataframe to Oracle database."""
    def __init__(self, name=None, filename=None, base_dir=None, layer_name=None, **kwargs):
        """Init."""
        super().__init__(name, filename, base_dir,  **kwargs)

    def save_dataset(self, dataset, table_name=None, user=None, password=None,
                     host=None, port=None, service_name=None, **kwargs):
        """Write data to Oracle database table."""
        write_gdf_to_oracle(dataset.data, table_name, user, password, host, port, service_name)

def create_multiline_geometry(multilines, t_obj, einfo_obj, otype_obj):
    """Creates an Oracle SDO_GEOMETRY object for a MultiLineString."""
    geometry = t_obj.newobject()
    geometry.SDO_GTYPE = 2006  # 2006 = MULTILINESTRING
    geometry.SDO_SRID = 4326   # SRID for WGS 84
    geometry.SDO_ELEM_INFO = einfo_obj.newobject()
    geometry.SDO_ORDINATES = otype_obj.newobject()

    elem_info = []
    ordinates = []
    start_index = 1

    for line in multilines.geoms:
        line_coords = [coord for point in line.coords for coord in point]
        elem_info.extend([start_index, 2, 1])  # Start index, LineString, Simple Element
        ordinates.extend(line_coords)
        start_index += len(line_coords)  # Update start index for next line

    geometry.SDO_ELEM_INFO.extend(elem_info)
    geometry.SDO_ORDINATES.extend(ordinates)

    return geometry

def write_gdf_to_oracle(gdf, table_name, user, password, host, port, service_name):
    """Write geodataframe to oracle database.

    The connection data like user/pw/host/port/service_name are taken from the config.

    Args:
        gdf (geopandas.GeoDataFrame): Dataframe with flash geometry data
        table_name (str): Name of the table in the oracle databse to write to
        user (str): Database user.
        password (str): Database password.
        host (str): Database host.
        port (int): Database port.
        service_name (str): Service name of database.
    """
    # Create Oracle DSN (Data Source Name)
    dsn = f"{host}:{port}/{service_name}"

    with oracledb.connect(user=user, password=password, dsn=dsn) as connection:
        cursor = connection.cursor()

        # Acquire types used for creating SDO_GEOMETRY objects
        type_obj = connection.gettype("MDSYS.SDO_GEOMETRY")
        element_info_type_obj = connection.gettype("MDSYS.SDO_ELEM_INFO_ARRAY")
        ordinate_type_obj = connection.gettype("MDSYS.SDO_ORDINATE_ARRAY")

        # Prepare the SQL insert statement
        insert_sql = f"""
            INSERT INTO {table_name} (flash_id, group_end_time, normalized_group_time, geom)
            VALUES (:flash_id, :group_end_time, :normalized_group_time, :obj)
        """

        # Prepare data for batch insertion
        data = []
        for i in range(len(gdf)):
            flash_id = int(gdf["flash_id"].iloc[i])
            group_end_time = gdf["group_end_time"].iloc[i]
            normalized_group_time = gdf["normalized_group_time"].iloc[i]
            geometry = gdf["geometry"].iloc[i]

            if isinstance(geometry, shapely.LineString):
                # Convert LineString to MultiLineString for consistency
                geometry = shapely.MultiLineString([geometry])

            if isinstance(geometry, shapely.MultiLineString):
                geom = create_multiline_geometry(geometry, t_obj=type_obj,
                                                 einfo_obj=element_info_type_obj,
                                                 otype_obj=ordinate_type_obj)

                # Append row as a tuple for executemany
                data.append((
                    flash_id,
                    group_end_time,
                    normalized_group_time,
                    geom
                ))

        # Execute batch insert
        cursor.executemany(insert_sql, data)

        # Commit the transaction
        connection.commit()
