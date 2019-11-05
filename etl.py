import configparser
from datetime import datetime
import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, dayofweek
from pyspark.sql.functions import to_timestamp, monotonically_increasing_id

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('AWS', 'AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('AWS', 'AWS_SECRET_ACCESS_KEY')

# Create logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s  [%(name)s] %(message)s')
logger = logging.getLogger('pyspark_etl')

def create_spark_session():
    """
    Create a spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Extract song and artist information from song data and save them to S3.
    """

    # get filepath to song data file
    song_data = "song_data/*/*/*/*.json"
    
    # read song data file
    logger.info('Reading song data files')
    df = spark.read.json(input_data + song_data)
    
    logger.info('Creating the staging_song view from song data')
    df.createOrReplaceTempView("staging_song")

    # extract columns to create songs table
    logger.info('Creating the songs table')
    songs_table = spark.sql("""
    select 
        song_id, 
        title, 
        artist_id, 
        case when year != 0 then year else null end as year,
        duration 
    from staging_song
    """)
    
    # write songs table to parquet files partitioned by year and artist
    logger.info('Writing songs.parquet')
    songs_table.write.partitionBy("year", "artist_id").parquet(output_data + "songs.parquet")

    # extract columns to create artists table
    logger.info('Creating the artists table')
    artists_table = spark.sql("""
    select 
        artist_id,
        name,
        location,
        latitude,
        longitude
    from (
        select 
            artist_id, 
            artist_name as name,
            artist_location as location,
            artist_latitude as latitude,
            artist_longitude as longitude,
            row_number() over (partition by artist_id order by year desc) as row_number
        from staging_song
    ) as artist
    where row_number = 1
    """)
    
    # write artists table to parquet files
    logger.info('Writing artists.parquet')
    artists_table.write.parquet(output_data + "artists.parquet")


def process_log_data(spark, input_data, output_data):
    """
    Extract user, time, and songplay information from log data and save them to S3.
    """

    # get filepath to log data file
    log_data = "log_data/*/*/*.json"

    # read log data file
    logger.info('Reading log data files')
    df = spark.read.json(input_data + log_data)
    
    # filter by actions for song plays
    logger.info('Filtering by actions for song plays')
    df = df.filter(df.page == 'NextSong')
    
    logger.info('Creating the staging_event view from log data')
    df.createOrReplaceTempView("staging_event")

    # extract columns for users table
    logger.info('Creating the users table')
    users_table = spark.sql("""
    select 
        user_id, 
        first_name,
        last_name,
        gender,
        level
    from (
        select 
            userId as user_id, 
            firstName as first_name,
            lastName as last_name,
            gender,
            level,
            row_number() over (partition by userId order by ts desc) as row_number
        from staging_event
    ) as user
    where row_number = 1
    """)
    
    # write users table to parquet files
    logger.info('Writing users.parquet')
    users_table.write.parquet(output_data + 'users.parquet')

    # create timestamp column and songplay_id column
    logger.info('Adding start_time and songplay_id columns')
    df = df \
        .withColumn("start_time", to_timestamp(df.ts/1000)) \
        .withColumn('songplay_id', monotonically_increasing_id())
    
    # Refresh the staging_event view to include new columns
    logger.info('Creating the staging_event view from log data')
    df.createOrReplaceTempView("staging_event")
    
    # extract columns to create time table
    logger.info('Creating the time table')
    time_table = time_table = df.select('start_time') \
        .dropDuplicates() \
        .withColumn('hour', hour('start_time')) \
        .withColumn('day', dayofmonth('start_time')) \
        .withColumn('week', weekofyear('start_time')) \
        .withColumn('month', month('start_time')) \
        .withColumn('year', year('start_time')) \
        .withColumn('weekday', dayofweek('start_time'))
    
    # write time table to parquet files partitioned by year and month
    logger.info('Writing time.parquet')
    time_table.write.partitionBy('year', 'month').parquet(output_data + 'time.parquet')

    # read in time data to use for songplays table
    logger.info('Creating the event_time view from the time table')
    time_table.createOrReplaceTempView("event_time")
    
    # extract columns from joined song and log datasets to create songplays table
    logger.info('Creating the songplays table')
    songplays_table = spark.sql("""
    select 
        se.songplay_id,
        se.start_time,
        se.userId as user_id,
        se.level,
        ss.song_id,
        ss.artist_id,
        se.sessionId as session_id,
        se.location,
        se.userAgent,
        et.year,
        et.month
    from staging_event se
    inner join staging_song ss on se.song = ss.title and se.artist = ss.artist_name and se.length = ss.duration
    inner join event_time et on se.start_time = et.start_time
    """)

    # write songplays table to parquet files partitioned by year and month
    logger.info('Writing songplays.parquet')
    songplays_table.write.partitionBy('year', 'month').parquet(output_data + 'songplays.parquet')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://yura.udacity.dend4.output/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
