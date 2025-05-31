import sqlite3
from flask import Flask, render_template, g, request, jsonify
from datetime import datetime, date, timedelta, time
from dateutil.parser import isoparse
import os
import calendar
import math
import subprocess
import pytz
from urllib.parse import unquote

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, 'train_journeys.db') # Assumes DB is in the same dir as app.py
# Path to the Node.js script
GENERATE_URL_SCRIPT = os.path.join(BASE_DIR, 'generate_url.js')
# Assuming 'node' is in your system's PATH
NODE_EXECUTABLE = 'node'


# --- Flask App Setup ---
app = Flask(__name__)
app.config['DATABASE'] = DATABASE

# --- Database Connection Handling ---
def get_db():
    """Gets a database connection for the current request."""
    db = getattr(g, '_database', None)
    if db is None:
        # Check if the database file exists before connecting
        if not os.path.exists(app.config['DATABASE']):
            print(f"Error: Database file not found at {app.config['DATABASE']}")
            return None # Return None if DB file doesn't exist

        db = g._database = sqlite3.connect(app.config['DATABASE'])
        db.row_factory = sqlite3.Row # Return rows as objects that behave like dicts
    return db

@app.teardown_appcontext
def close_db(error):
    """Closess the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# --- Helper for formatting ---
def format_duration(minutes):
    """Converts minutes to HH:MM format."""
    if minutes is None:
        return "N/A"
    try:
        minutes = int(minutes)
    except (ValueError, TypeError):
         return "Invalid Duration"

    if minutes < 0:
        return "N/A (Invalid duration)"
    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours:02d}h {remaining_minutes:02d}m"

def format_datetime(iso_string):
    """Parses ISO 8601 string and formats it nicely."""
    if iso_string is None:
        return "N/A"
    try:
        dt_object = isoparse(iso_string)
        return dt_object.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return iso_string

def format_price(amount, currency):
    """Formats price amount and currency."""
    if amount is None:
         return "N/A"
    try:
        formatted_amount = f"{amount:.2f}"
        if currency:
             currency_symbol = {
                 'EUR': '€',
                 'USD': '$',
                 'GBP': '£'
             }.get(str(currency).upper(), str(currency))
             return f"{formatted_amount} {currency_symbol}"
        else:
             return formatted_amount
    except Exception:
         return str(amount)

# --- Helper for calendar date logic ---
def add_months(sourcedate, months):
    """Adds months to a date, handling month and year rollovers."""
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return date(year, month, day)

# --- Helper function to parse HH:MM duration string to minutes ---
def parse_hh_mm_to_minutes(hh_mm_str):
    """
    Parses a string in HH:MM format into a total number of minutes.
    Returns integer minutes or None if invalid format.
    """
    if not hh_mm_str:
        return None
    try:
        parts = hh_mm_str.split(':')
        if len(parts) == 2:
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours * 60 + minutes
        else:
            return None # Invalid format
    except ValueError:
        return None # Invalid numbers

# --- Helper function to parse and adjust HH:MM time string by subtracting hours ---
def adjust_hh_mm_by_hours(hh_mm_str, hours_to_subtract):
    """
    Parses a string in HH:MM format, subtracts the specified hours,
    and returns the result as an HH:MM string. Handles wrapping around midnight.
    Returns the original string if parsing fails.
    """
    if not hh_mm_str:
        return hh_mm_str

    try:
        # Parse the time string
        time_obj = datetime.strptime(hh_mm_str, '%H:%M').time()
        # Combine with a dummy date to use timedelta
        dummy_datetime = datetime.combine(date(2000, 1, 1), time_obj)

        # Subtract the hours
        adjusted_datetime = dummy_datetime - timedelta(hours=hours_to_subtract)

        # Format the result back to HH:MM
        return adjusted_datetime.strftime('%H:%M')

    except ValueError:
        # If parsing fails, return the original string and log a warning
        print(f"Warning: Could not parse time string '{hh_mm_str}' for adjustment. Using original.")
        return hh_mm_str
    except Exception as e:
         print(f"An unexpected error occurred during time adjustment: {e}. Using original string.")
         return hh_mm_str


# --- Helper function to build time filter clause ---
def build_time_filter_clause(filter_type, start_time, end_time, column_prefix='j.'):
    """
    Builds the SQL time filter clause and parameters.
    filter_type: 'departure', 'arrival', or ''
    start_time: HH:MM string (already adjusted if needed)
    end_time: HH:MM string (already adjusted if needed)
    column_prefix: Prefix for column names (e.g., 'j.')
    Returns a tuple (clause, params)
    """
    clause = ""
    params = []
    time_column_to_filter = None

    if filter_type == 'departure':
        time_column_to_filter = f'{column_prefix}start_time'
    elif filter_type == 'arrival':
        time_column_to_filter = f'{column_prefix}end_time'

    # Use the potentially adjusted start_time and end_time directly
    start_time_formatted = start_time.strip()
    end_time_formatted = end_time.strip()


    if time_column_to_filter and start_time_formatted and end_time_formatted:
        # Basic validation: ensure format is somewhat correct (HH:MM)
        # Note: We adjusted the times, so the format should be okay if original was HH:MM
        if len(start_time_formatted) == 5 and ':' in start_time_formatted and \
           len(end_time_formatted) == 5 and ':' in end_time_formatted:

            # Handle time ranges that cross midnight (e.g., 22:00 to 04:00)
            # SQLite time() function helps here, compares HH:MM:SS strings
            # We'll use HH:MM from the input, implicitly assuming seconds are 00
            if start_time_formatted <= end_time_formatted:
                 # Simple range within the same day
                 clause = f" AND strftime('%H:%M', {time_column_to_filter}) BETWEEN ? AND ?"
                 params = [start_time_formatted, end_time_formatted]
            else:
                 # Range crosses midnight (e.g., 22:00 to 04:00)
                 # WHERE time >= '22:00' OR time <= '04:00'
                 clause = f" AND (strftime('%H:%M', {time_column_to_filter}) >= ? OR strftime('%H:%M', {time_column_to_filter}) <= ?)"
                 params = [start_time_formatted, end_time_formatted]
        else:
             # This case should be rare if adjust_hh_mm_by_hours works, but keeping for safety
             print(f"Warning: Adjusted time format became invalid. Ignoring time filter.")
             # Return empty clause and params
             clause = ""
             params = []


    return clause, params


# --- New Endpoint to Generate and Fetch Journey URL ---
@app.route('/generate_journey_url/<int:journey_id>')
def generate_journey_url_route(journey_id):
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database file not found.'}), 500

    cursor = db.cursor()

    try:
        # Fetch the journey details
        cursor.execute("SELECT journey_id, url, refreshToken FROM journeys WHERE journey_id = ?", (journey_id,))
        journey = cursor.fetchone()

        if not journey:
            return jsonify({'error': 'Journey not found.'}), 404

        existing_url = journey['url']
        refresh_token = journey['refreshToken']

        if existing_url:
            # URL already exists, return it
            print(f"URL already exists for journey {journey_id}: {existing_url}")
            return jsonify({'url': existing_url})

        elif refresh_token:
            # URL is null, but refreshToken exists, generate the URL
            print(f"Generating URL for journey {journey_id} using refreshToken: {refresh_token}")
            try:
                # Check if the Node.js script exists
                if not os.path.exists(GENERATE_URL_SCRIPT):
                    print(f"Error: Node.js script not found at {GENERATE_URL_SCRIPT}")
                    return jsonify({'error': 'Server configuration error: URL generator script not found.'}), 500

                # Execute the Node.js script
                # Assumes 'node' command is available in the environment's PATH
                process_args = [NODE_EXECUTABLE, GENERATE_URL_SCRIPT, refresh_token]
                # Optional: Add language, e.g., process_args = [NODE_EXECUTABLE, GENERATE_URL_SCRIPT, refresh_token, 'en']
                # For now, sticking to script's default 'de' or provided language

                result = subprocess.run(
                    process_args,
                    capture_output=True,
                    text=True,
                    check=True, # Raise CalledProcessError if script returns non-zero exit code
                    cwd=BASE_DIR # Run the script from the base directory
                )

                generated_url = result.stdout.strip()
                print(f"Node.js script output: {generated_url}")

                if generated_url:
                    # Save the generated URL back to the database
                    cursor.execute("UPDATE journeys SET url = ? WHERE journey_id = ?", (generated_url, journey_id))
                    db.commit()
                    print(f"Saved generated URL for journey {journey_id}.")
                    return jsonify({'url': generated_url})
                else:
                    # Script ran but produced no URL
                    print(f"Node.js script produced no URL for journey {journey_id}. Stderr: {result.stderr.strip()}")
                    return jsonify({'error': 'URL generation failed (script output empty).'}), 500

            except FileNotFoundError:
                 print(f"Error: Node.js executable ('{NODE_EXECUTABLE}') not found. Is Node.js installed and in PATH?")
                 return jsonify({'error': 'Server configuration error: Node.js not found.'}), 500
            except subprocess.CalledProcessError as e:
                print(f"Error running Node.js script for journey {journey_id}: {e}")
                print(f"Script stderr: {e.stderr.strip()}")
                print(f"Script stdout: {e.stdout.strip()}")
                return jsonify({'error': f'URL generation script failed: {e.stderr.strip()}'}), 500
            except Exception as e:
                print(f"An unexpected error occurred during URL generation for journey {journey_id}: {e}")
                return jsonify({'error': f'An internal error occurred: {str(e)}'}), 500

        else:
            # Neither URL nor refreshToken is available
            print(f"No URL or refreshToken for journey {journey_id}.")
            return jsonify({'error': 'No information available to generate a link for this journey.'}), 400 # Bad request because no token

    except sqlite3.Error as e:
        print(f"Database error fetching or updating journey {journey_id}: {e}")
        db.rollback() # Rollback any partial updates
        return jsonify({'error': f'Database error: {e}'}), 500
    except Exception as e:
        print(f"An unexpected server error occurred for journey {journey_id}: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500





@app.route('/api/journeys/<date>')
def get_journeys_by_date(date):
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database file not found.'}), 500

    cursor = db.cursor()
    try:
        selected_origin = unquote(request.args.get('origin', ''))
        selected_destination = unquote(request.args.get('destination', ''))
        print(f"Origin: '{selected_origin}', Destination: '{selected_destination}'")

        if request.args.get('round_trip') == 'on':
            day_count = int(request.args.get('day_count', 1))
            exclude_partial = 1 if request.args.get('exclude_partial_fare') == 'on' else 0
            duration_limit = parse_hh_mm_to_minutes(request.args.get('exclude_duration_over')) or -1

            def adjust_time(time_str, hours):
                try:
                    h, m = map(int, time_str.split(':'))
                    adj_h = (h + hours) % 24
                    return f"{adj_h:02d}:{m:02d}"
                except:
                    return time_str

            outbound_start = adjust_time(request.args.get('outbound_start_time', '00:00'), 2)
            outbound_end = adjust_time(request.args.get('outbound_end_time', '23:59'), 2)
            return_start = adjust_time(request.args.get('return_start_time', '00:00'), 2)
            return_end = adjust_time(request.args.get('return_end_time', '23:59'), 2)

            outbound_query = f'''
                SELECT *,
                    DATE(start_time, '+{day_count} days') as return_date 
                FROM journeys
                WHERE DATE(start_time) = ?
                AND origin_name = ?
                AND destination_name = ?
                AND ((TIME(start_time) BETWEEN ? AND ?) OR (? > ? AND (TIME(start_time) >= ? OR TIME(start_time) <= ?)))
                AND ({exclude_partial} = 0 OR partialFare != 1)
                AND ({duration_limit} = -1 OR duration_minutes <= ?)
                ORDER BY price_amount ASC
            '''
            outbound_params = (
                date, selected_origin, selected_destination, outbound_start, outbound_end,
                outbound_start, outbound_end, outbound_start, outbound_end, duration_limit
            )

            cursor.execute(outbound_query, outbound_params)
            outbound_journeys = cursor.fetchall()

            round_trip_pairs = []
            if outbound_journeys:
                return_dates = [row['return_date'] for row in outbound_journeys if 'return_date' in row.keys()]
                print(f"Looking for returns on: {return_dates}")

                if return_dates:
                    return_query = f'''
                        SELECT *, DATE(start_time) as return_date 
                        FROM journeys
                        WHERE DATE(start_time) IN ({','.join(['?']*len(return_dates))})
                        AND origin_name = ?
                        AND destination_name = ?
                        AND ((TIME(start_time) BETWEEN ? AND ?) OR (? > ? AND (TIME(start_time) >= ? OR TIME(start_time) <= ?)))
                        AND ({exclude_partial} = 0 OR partialFare != 1)
                        AND ({duration_limit} = -1 OR duration_minutes <= ?)
                        ORDER BY price_amount ASC
                    '''
                    return_params = return_dates + [
                        selected_destination, selected_origin, return_start, return_end,
                        return_start, return_end, return_start, return_end, duration_limit
                    ]

                    cursor.execute(return_query, return_params)
                    # Ensure return journeys are grouped by date and sorted by price
                    returns = {}
                    for row in cursor.fetchall():
                        return_date = row['return_date']
                        if return_date not in returns:
                            returns[return_date] = []
                        returns[return_date].append(dict(row))
                    
                    # Sort each date's return journeys by price
                    for return_date in returns:
                        returns[return_date].sort(key=lambda x: x['price_amount'] if x['price_amount'] is not None else float('inf'))

                    for outbound in outbound_journeys:
                        return_date = outbound['return_date'] if 'return_date' in outbound.keys() else None
                        if return_date and return_date in returns and returns[return_date]:
                            # Pick the cheapest return journey for the date
                            cheapest_return = returns[return_date][0]
                            round_trip_pairs.append({
                                'outbound': dict(outbound),
                                'return': cheapest_return,
                                'total_price': outbound['price_amount'] + cheapest_return['price_amount']
                            })

            return jsonify({'round_trip_pairs': round_trip_pairs})

        else:
            exclude_partial = 1 if request.args.get('exclude_partial_fare') == 'on' else 0
            duration_limit = parse_hh_mm_to_minutes(request.args.get('exclude_duration_over')) or -1

            query = '''
                SELECT journey_id, origin_name, destination_name,
                       start_time, end_time, duration_minutes,
                       price_amount, price_currency, url,
                       refreshToken, partialFare
                FROM journeys
                WHERE DATE(start_time) = ?
                AND origin_name = ?
                AND destination_name = ?
                AND (? = 0 OR partialFare != 1)
                AND (? = -1 OR duration_minutes <= ?)
                ORDER BY price_amount ASC
            '''
            params = [
                date,
                selected_origin,
                selected_destination,
                exclude_partial,
                duration_limit,
                duration_limit
            ]

            print("One-way query:", query)
            print("One-way params:", params)
            cursor.execute(query, params)
            journeys = cursor.fetchall()
            print(f"Found {len(journeys)} one-way journeys")

            formatted_journeys = [dict(row) for row in journeys]
            return jsonify({'journeys': formatted_journeys})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500



# --- Main Index Route ---
@app.route('/')
def index():
    db = get_db()
    if db is None:
        # Handle case where database file is missing
        return render_template('index.html', error="Database file not found.")

    cursor = db.cursor()
    today_str = date.today().isoformat()  # Added for date filtering
    today_date = date.today()  # Added for calendar filtering

    # save requests in database

    selected_origin = request.args.get('origin', '')
    selected_destination = request.args.get('destination', '')

    # --- Save the search parameters to the requested_journeys table ---
    if selected_origin and selected_destination:
        try:
            # Get the current time in the local timezone
            local_timezone = pytz.timezone("Europe/Amsterdam")  # Replace with your timezone
            local_time = datetime.now(local_timezone)

            # Insert the search parameters with the correct timestamp
            cursor.execute("""
                INSERT INTO requested_journeys (origin_name, destination_name, requested_at)
                VALUES (?, ?, ?)
            """, (selected_origin, selected_destination, local_time))
            db.commit()
            print(f"Search logged: Origin='{selected_origin}', Destination='{selected_destination}', Time='{local_time}'")
        except sqlite3.Error as e:
            print(f"Error logging search to 'requested_journeys': {e}")





    # --- Get unique Origin, Destination, and Processed Dates for dropdowns ---
    # These are always fetched regardless of filters to populate dropdowns
    # Initialize lists in case of query errors
    unique_origins = []
    unique_destinations = []
    unique_processed_dates_str = []
    available_stations = []
    try:
        cursor.execute("""
            SELECT DISTINCT origin_name
            FROM journeys
            WHERE origin_name IS NOT NULL AND origin_name != ''
            ORDER BY origin_name;
        """)
        unique_origins = [row['origin_name'] for row in cursor.fetchall()]

        cursor.execute("""
            SELECT DISTINCT destination_name
            FROM journeys
            WHERE destination_name IS NOT NULL AND destination_name != ''
            ORDER BY destination_name;
        """)
        unique_destinations = [row['destination_name'] for row in cursor.fetchall()]

        # Get unique processed dates as-MM-DD strings
        cursor.execute("""
            SELECT DISTINCT strftime('%Y-%m-%d', processed_at) AS processed_date_str
            FROM requests
            WHERE processed_at IS NOT NULL AND processed_at != ''
            ORDER BY processed_date_str DESC; -- Newest dates first
        """)
        unique_processed_dates_str = [row['processed_date_str'] for row in cursor.fetchall()]

        # Fetch available stations from the select_stations table
        cursor.execute("SELECT DISTINCT name FROM select_stations ORDER BY name;")
        available_stations = [row['name'] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        print(f"Database error fetching unique values: {e}")
        # Continue with empty lists, render_template will show no data if main query also fails/returns empty


    # --- Get filter selections from the request ---
    exclude_past_journeys = request.args.get('exclude_past', '') == 'on'
    selected_origin = request.args.get('origin', '')
    selected_destination = request.args.get('destination', '')
    selected_processed_date = request.args.get('processed_date', '')
    selected_day_count = request.args.get('day_count', type=int, default=1) # Get day count, default 1
    selected_round_trip_mode = request.args.get('round_trip', '') == 'on' # Check if toggle is 'on'



    # Outbound Time filter parameters (Original input strings)
    original_outbound_start_time = request.args.get('outbound_start_time', '') # HH:MM string
    original_outbound_end_time = request.args.get('outbound_end_time', '')   # HH:MM string
    selected_time_filter_type = request.args.get('time_filter_type', '') # 'departure', 'arrival', or ''

    # Return Time filter parameters (Original input strings)
    original_return_start_time = request.args.get('return_start_time', '') # HH:MM string
    original_return_end_time = request.args.get('return_end_time', '')   # HH:MM string
    selected_return_time_filter_type = request.args.get('return_time_filter_type', '') # 'departure', 'arrival', or ''


    # Apply -2 hour offset to time inputs for filtering
    # These are the times used in the SQL query
    adjusted_outbound_start_time = original_outbound_start_time # Default to original if no type selected
    adjusted_outbound_end_time = original_outbound_end_time   # Default to original if no type selected
    if selected_time_filter_type: # Only apply offset if a time filter type is selected
        adjusted_outbound_start_time = adjust_hh_mm_by_hours(original_outbound_start_time, 2)
        adjusted_outbound_end_time = adjust_hh_mm_by_hours(original_outbound_end_time, 2)


    adjusted_return_start_time = original_return_start_time # Default to original if no type selected
    adjusted_return_end_time = original_return_end_time   # Default to original if no type selected
    if selected_return_time_filter_type: # Only apply offset if a return time filter type is selected
        adjusted_return_start_time = adjust_hh_mm_by_hours(original_return_start_time, 2)
        adjusted_return_end_time = adjust_hh_mm_by_hours(original_return_end_time, 2)


    # Partial Fare filter parameter
    exclude_partial_fare = request.args.get('exclude_partial_fare', '') == 'on'

    # Highlight Price parameter
    selected_highlight_price_under = request.args.get('highlight_price_under', type=float)
    # Note: type=float will return None if the input is empty or not a valid float

    # Exclude Duration filter parameter (Original input string and parsed minutes)
    selected_exclude_duration_over = request.args.get('exclude_duration_over', '') # HH:MM string
    exclude_duration_over_minutes = parse_hh_mm_to_minutes(selected_exclude_duration_over)


    # Ensure day count is not negative
    if selected_day_count is None or selected_day_count < 0: # Handle potential None from type=int if input is empty/invalid
        selected_day_count = 1

    # --- Initialize variables that will be used later and passed to template ---
    # Explicitly initialize them here to guarantee they are bound in all paths
    # In RT mode, this will hold pairs of (outbound, return) journeys
    # In one-way mode, this will hold lists of one-way journeys as before
    journeys_for_detailed_view = {}

    # Temporary structure to hold outbound journeys grouped by day, used for RT calendar base
    journeys_by_day_outbound_temp = {} # *** Initialize here ***

    cheapest_prices_for_calendar_raw = {} # Raw prices (one-way or RT) for calendar (keyed by date)
    cheapest_prices_for_calendar_formatted = {} # Formatted prices (one-way or RT) for calendar (keyed by date)
    monthly_calendars = []               # Calendar structure data (list of month dicts)
    all_dates_with_data = set()          # Set of date objects that should appear in calendar


    # Day names for the calendar header - initialize outside conditional logic
    calendar.setfirstweekday(calendar.MONDAY)
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # --- Build Time Filter SQL Clauses for Outbound and Return ---
    # Use the helper function with the ADJUSTED times
    outbound_time_filter_clause, outbound_time_filter_params = build_time_filter_clause(
        selected_time_filter_type, adjusted_outbound_start_time, adjusted_outbound_end_time
    )

    return_time_filter_clause, return_time_filter_params = build_time_filter_clause(
         selected_return_time_filter_type, adjusted_return_start_time, adjusted_return_end_time
    )

    # --- Build Partial Fare Filter SQL Clause ---
    partial_fare_filter_clause = ""
    if exclude_partial_fare:
        partial_fare_filter_clause = " AND j.partialFare != 1"

    # --- Build Duration Filter SQL Clause ---
    duration_filter_clause = ""
    duration_filter_params = []
    if exclude_duration_over_minutes is not None:
        duration_filter_clause = " AND j.duration_minutes < ?"
        duration_filter_params.append(exclude_duration_over_minutes)


    # --- Data Processing based on Round Trip Mode ---

    # Only attempt round trip search if essential parameters are selected
    if selected_round_trip_mode and selected_origin and selected_destination and selected_origin != selected_destination:
        print(f"Round Trip Search: {selected_origin} to {selected_destination} with {selected_day_count} nights")
        print(f"Outbound Time Filter (Input): Type={selected_time_filter_type}, Range={original_outbound_start_time}-{original_outbound_end_time}")
        print(f"Outbound Time Filter (Adjusted): Range={adjusted_outbound_start_time}-{adjusted_outbound_end_time}")
        print(f"Return Time Filter (Input): Type={selected_return_time_filter_type}, Range={original_return_start_time}-{original_return_end_time}")
        print(f"Return Time Filter (Adjusted): Range={adjusted_return_start_time}-{adjusted_return_end_time}")
        print(f"Exclude Partial Fare: {exclude_partial_fare}")
        print(f"Exclude Duration Over: {selected_exclude_duration_over} ({exclude_duration_over_minutes} mins)")


        if selected_round_trip_mode and selected_origin and selected_destination:
            # 1. Fetch potential Outbound Journeys (Origin -> Destination)
            outbound_query = """
                SELECT
                    j.journey_id,
                    j.origin_name,
                    j.destination_name,
                    j.start_time,
                    j.end_time,
                    j.duration_minutes,
                    j.price_amount,
                    j.price_currency,
                    j.url,
                    j.refreshToken,
                    j.partialFare
                FROM journeys j
                JOIN requests r ON j.request_id = r.request_id
                WHERE j.origin_name = ?
                AND j.destination_name = ?
                AND j.start_time IS NOT NULL AND j.start_time != ''
            """
            outbound_params = [selected_origin, selected_destination]

            # In the one-way query section:
            if exclude_past_journeys:
                outbound_query += " AND DATE(j.start_time) >= ?"
                outbound_params.append(today_str)  # Add this line


            # Apply the OUTBOUND TIME FILTER (using adjusted times)
            outbound_query += outbound_time_filter_clause
            outbound_params.extend(outbound_time_filter_params)

            # Apply the PARTIAL FARE FILTER to outbound
            outbound_query += partial_fare_filter_clause

            # Apply the DURATION FILTER to outbound
            outbound_query += duration_filter_clause
            outbound_params.extend(duration_filter_params)

            outbound_query += " ORDER BY j.start_time ASC;"

            outbound_journeys = []
            try:
                cursor.execute(outbound_query, outbound_params)
                outbound_journeys = cursor.fetchall()
            except sqlite3.Error as e:
                print(f"Database error fetching outbound journeys with filters: {e}")
                # outbound_journeys remains empty list

            # Group outbound journeys by day for the detailed view base and RT calendar base
            # In RT mode, detailed view shows pairs, but this grouping is needed to check days with *any* potential outbound
            for journey in outbound_journeys:
                start_time_str = journey['start_time']
                try:
                    dt_object = isoparse(start_time_str)
                    date_key = dt_object.date().isoformat()
                    if date_key not in journeys_by_day_outbound_temp: # Populate the shared temp dict
                        journeys_by_day_outbound_temp[date_key] = []
                    journeys_by_day_outbound_temp[date_key].append(journey)
                except Exception as e:
                    print(f"Warning: Could not parse start_time for outbound journey {journey.get('journey_id', 'N/A')}: {e}. Skipping for grouping.")
                    continue


                        # Post-query filtering for past journeys (NEW)
            if exclude_past_journeys:
                journeys_by_day_outbound_temp = {
                    k: v for k, v in journeys_by_day_outbound_temp.items()
                    if isoparse(k).date() >= today_date
                }


            # 2. Fetch potential Return Journeys (Destination -> Origin)
            return_query = """
                SELECT
                    j.journey_id,
                    j.origin_name,
                    j.destination_name,
                    j.start_time,
                    j.end_time,
                    j.duration_minutes,
                    j.price_amount,
                    j.price_currency,
                    j.url,
                    j.refreshToken,
                    j.partialFare
                FROM journeys j
                JOIN requests r ON j.request_id = r.request_id
                WHERE j.origin_name = ?
                AND j.destination_name = ?
            """
            return_params = [selected_destination, selected_origin]
            if selected_processed_date:
                return_query += " AND strftime('%Y-%m-%d', r.processed_at) = ?"
                return_params.append(selected_processed_date)
            
            # Add time filter clause BEFORE date filter
            return_query += return_time_filter_clause
            return_params.extend(return_time_filter_params)
            
            if exclude_past_journeys:
                return_query += " AND DATE (j.start_time) >= ?"
                return_params.append(today_str)
        

        # Apply the PARTIAL FARE FILTER to return
        return_query += partial_fare_filter_clause

         # Apply the DURATION FILTER to return
        return_query += duration_filter_clause
        return_params.extend(duration_filter_params)


        return_query += " ORDER BY j.start_time ASC;" # Order by start time is fine, but not strictly needed for exact date grouping

        return_journeys = []
        try:
            cursor.execute(return_query, return_params)
            return_journeys = cursor.fetchall()
        except sqlite3.Error as e:
             print(f"Database error fetching return journeys with filters: {e}")
             # return_journeys remains empty list


        # 3. Group and Sort Return Journeys by EXACT Departure Date
        # This creates a structure to quickly look up return journeys on a specific date
        sorted_return_journeys_by_date = {} # {'YYYY-MM-DD': [sorted list of returns on this date], ...}
        return_journeys_by_date_temp = {} # Intermediate grouping

        for return_j in return_journeys:
             return_date_str = None
             try:
                  return_dt = isoparse(return_j['start_time'])
                  return_date_str = return_dt.date().isoformat()
             except Exception:
                    try:
                        journey_id = return_j['journey_id']
                    except KeyError:
                        journey_id = 'N/A'
                    continue

             if return_date_str not in return_journeys_by_date_temp:
                  return_journeys_by_date_temp[return_date_str] = []
             return_journeys_by_date_temp[return_date_str].append(return_j)

        # Now sort the journeys within each date by price
        for date_key in return_journeys_by_date_temp.keys():
             day_returns = return_journeys_by_date_temp[date_key]
             day_returns.sort(key=lambda j: j['price_amount'] if j['price_amount'] is not None else math.inf)
             sorted_return_journeys_by_date[date_key] = day_returns # Store the sorted list for this exact date


        # 4. Match outbound journeys with the cheapest eligible return on the EXACT date
        #    and prepare data for detailed view (pairs) and calendar (combined price)
        cheapest_round_trip_prices = {} # Calculate RT prices here {'outbound_date_str': {'amount': price, 'currency': currency}, ...}
        # New structure for detailed view in RT mode
        round_trip_pairs_by_outbound_day = {} # {'outbound_date_str': [(outbound_j, return_j), ...]}


        for outbound_j in outbound_journeys:
            outbound_date_str = None
            outbound_price = outbound_j['price_amount'] if outbound_j['price_amount'] is not None else math.inf
            if outbound_price == math.inf: # Cannot form a round trip without outbound price
                 continue

            try:
                 outbound_dt = isoparse(outbound_j['start_time'])
                 outbound_date_str = outbound_dt.date().isoformat()
                 # Calculate the EXACT required return date
                 required_return_date_obj = outbound_dt.date() + timedelta(days=selected_day_count)
                 required_return_date_str = required_return_date_obj.isoformat()
            except Exception as e:
                 print(f"Warning: Could not calculate return date for outbound journey {outbound_j.get('journey_id', 'N/A')}: {e}. Skipping.")
                 continue

            # Look up return journeys ONLY on the required_return_date in the sorted structure
            # These return journeys have already been filtered by the return time filter and partial fare filter if specified
            # They are also filtered by the duration filter
            eligible_return_journeys_on_date = sorted_return_journeys_by_date.get(required_return_date_str)

            cheapest_eligible_return_journey = None
            if eligible_return_journeys_on_date:
                 # The list is already sorted by price, so the first one is the cheapest for this exact date
                 cheapest_eligible_return_journey = eligible_return_journeys_on_date[0]


            if cheapest_eligible_return_journey:
                 return_price = cheapest_eligible_return_journey['price_amount'] if cheapest_eligible_return_journey['price_amount'] is not None else math.inf

                 if return_price != math.inf: # Ensure return also has a price
                      total_price = outbound_price + return_price

                      # --- Update Calendar Price ---
                      current_cheapest_round_trip_for_day_info = cheapest_round_trip_prices.get(outbound_date_str)

                      # Check if this is the cheapest round trip found so far for this outbound day
                      if (current_cheapest_round_trip_for_day_info is None or total_price < current_cheapest_round_trip_for_day_info['raw_amount']):
                         # Only update calendar price if this pair is cheaper for the day
                         cheapest_round_trip_prices[outbound_date_str] = {
                               'formatted': format_price(total_price, outbound_j['price_currency']), # Formatted price for display
                               'raw_amount': total_price # Raw numeric amount for comparison/highlighting
                           }


                      # --- Prepare Data for Detailed View (Store the pair) ---
                      if outbound_date_str not in round_trip_pairs_by_outbound_day:
                            round_trip_pairs_by_outbound_day[outbound_date_str] = []
                      # Store the pair of the current outbound journey and its cheapest return
                      round_trip_pairs_by_outbound_day[outbound_date_str].append((outbound_j, cheapest_eligible_return_journey))


        # Sort the pairs within each day for the detailed view by OUTBOUND PRICE
        journeys_for_detailed_view = {} # This will be passed to the template as journeys_by_day

        if round_trip_pairs_by_outbound_day:
            sorted_dates_strings = sorted(round_trip_pairs_by_outbound_day.keys())
            for date_key in sorted_dates_strings:
                pairs_on_day = round_trip_pairs_by_outbound_day[date_key]
                # Sort by the outbound journey's price_amount, handling None values
                pairs_on_day.sort(key=lambda pair: pair[0]['price_amount'] if pair[0]['price_amount'] is not None else math.inf) # <--- MODIFIED LINE
                journeys_for_detailed_view[date_key] = pairs_on_day # Store the sorted list of pairs


        # Ensure days with outbound journeys but no matching round trip still appear on the calendar
        # but without a price or link (or with an indicator).
        # Iterate over the dates that had outbound journeys (from journeys_by_day_outbound_temp)
        # These outbound journeys are already filtered by the outbound time filter and partial fare filter and duration filter
        for date_key in sorted(journeys_by_day_outbound_temp.keys()): # *** Use journeys_by_day_outbound_temp here ***
             if date_key not in cheapest_round_trip_prices: # Check the raw prices dict
                  # Add an entry indicating no RT found for this date in the calendar price data
                  cheapest_round_trip_prices[date_key] = {
                      'formatted': "N/A", # Formatted price for display
                      'raw_amount': None # No raw amount for comparison
                  }

        # Populate the formatted and raw prices for the calendar template from the consolidated RT prices
        cheapest_prices_for_calendar_formatted = {d: data['formatted'] for d, data in cheapest_round_trip_prices.items()}
        cheapest_prices_for_calendar_raw = {d: data['raw_amount'] for d, data in cheapest_round_trip_prices.items()}


        # Update all_dates_with_data based on dates that have calendar data (either RT or N/A placeholder)
        all_dates_with_data = set()
        try:
             # The keys of cheapest_round_trip_prices are the outbound dates we considered
             all_dates_with_data = {isoparse(d).date() for d in cheapest_round_trip_prices.keys()}
        except Exception as e:
             print(f"Error parsing dates for calendar generation from cheapest_round_trip_prices keys: {e}")
             # all_dates_with_data remains empty set


    else: # --- One-Way Trip Logic (Existing Logic + Filters) ---
        print("One-Way Search")
        print(f"Time Filter (Input): Type={selected_time_filter_type}, Range={original_outbound_start_time}-{original_outbound_end_time}") # Use outbound vars for one-way
        print(f"Time Filter (Adjusted): Range={adjusted_outbound_start_time}-{adjusted_outbound_end_time}")
        print(f"Exclude Partial Fare: {exclude_partial_fare}")
        print(f"Highlight Price Under: {selected_highlight_price_under}")
        print(f"Exclude Duration Over: {selected_exclude_duration_over} ({exclude_duration_over_minutes} mins)")


        # Build the main journey query with filters (Origin -> Destination)
        base_query = """
            SELECT
                j.journey_id,
                j.origin_name,
                j.destination_name,
                j.start_time,
                j.end_time,
                j.duration_minutes,
                j.price_amount,
                j.price_currency,
                j.url,
                j.refreshToken,
                j.partialFare
            FROM journeys j
            JOIN requests r ON j.request_id = r.request_id
            WHERE j.start_time IS NOT NULL AND j.start_time != '' -- Journeys must have a start time
        """
        where_clauses = []
        query_params = []

        if exclude_past_journeys:
            where_clauses.append("DATE(j.start_time) >= ?")
            query_params.append(today_str)




        if selected_origin:
            where_clauses.append("j.origin_name = ?")
            query_params.append(selected_origin)

        if selected_destination:
            where_clauses.append("j.destination_name = ?")
            query_params.append(selected_destination)

        if selected_processed_date:
            where_clauses.append("strftime('%Y-%m-%d', r.processed_at) = ?")
            query_params.append(selected_processed_date)

        # Apply the OUTBOUND TIME FILTER (used for one-way, using adjusted times)
        if outbound_time_filter_clause:
            # outbound_time_filter_clause already includes ' AND ' at the start
             where_clauses.append(outbound_time_filter_clause[len(" AND "):]) # Remove leading ' AND ' as we are joining clauses
             query_params.extend(outbound_time_filter_params)

        # Apply the PARTIAL FARE FILTER for one-way
        if partial_fare_filter_clause:
             where_clauses.append(partial_fare_filter_clause[len(" AND "):]) # Remove leading ' AND '
             # No additional params for partial fare filter

        # Apply the DURATION FILTER for one-way
        if duration_filter_clause:
             where_clauses.append(duration_filter_clause[len(" AND "):]) # Remove leading ' AND '
             query_params.extend(duration_filter_params)


        if where_clauses:
            full_query = base_query + " AND " + " AND ".join(where_clauses)
        else:
            full_query = base_query # Should technically always have the start_time IS NOT NULL clause now

        full_query += " ORDER BY j.start_time ASC;"

        journeys_data = []
        try:
            cursor.execute(full_query, query_params)
            journeys_data = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database error fetching one-way journeys with filters: {e}")
            # journeys_data remains empty list


        # Process Data for Display (grouping and cheapest price per day)
        journeys_by_day_one_way_temp = {} # Temp dict for grouping
        cheapest_one_way_prices_raw = {} # Calculate one-way raw prices here
        cheapest_one_way_prices_formatted = {} # Calculate one-way formatted prices here

        for journey in journeys_data:
            start_time_str = journey['start_time']
            try:
                dt_object = isoparse(start_time_str)
                date_obj = dt_object.date()
                date_key = date_obj.isoformat()

                if date_key not in journeys_by_day_one_way_temp:
                    journeys_by_day_one_way_temp[date_key] = []
                journeys_by_day_one_way_temp[date_key].append(journey)

            except Exception as e:
                 print(f"Warning: Could not parse start_time '{start_time_str}' for journey ID {journey.get('journey_id', 'N/A')}: {e}. Skipping journey.")
                 continue


        # Sort the one-way journeys within each day and find cheapest one-way price
        if journeys_by_day_one_way_temp:
            sorted_dates_strings = sorted(journeys_by_day_one_way_temp.keys())
            for date_key in sorted_dates_strings:
                 day_journeys = journeys_by_day_one_way_temp[date_key]
                 day_journeys.sort(key=lambda j: j['price_amount'] if j['price_amount'] is not None else math.inf)
                 journeys_for_detailed_view[date_key] = day_journeys # Assign to the outer variable for one-way


                 # Find the cheapest price for the day (from the now-sorted list)
                 if day_journeys and day_journeys[0]['price_amount'] is not None: # Ensure there's at least one journey and it has a price
                    cheapest_journey_today = journeys_for_detailed_view[date_key][0]
                    cheapest_price = cheapest_journey_today['price_amount']
                    cheapest_currency = cheapest_journey_today['price_currency']
                    cheapest_one_way_prices_formatted[date_key] = format_price(cheapest_price, cheapest_currency)
                    cheapest_one_way_prices_raw[date_key] = cheapest_price # Store raw price
                 else:
                     cheapest_one_way_prices_formatted[date_key] = "N/A" # Mark as N/A if no valid journey or no price
                     cheapest_one_way_prices_raw[date_key] = None # No raw price


            cheapest_prices_for_calendar_formatted = cheapest_one_way_prices_formatted # Calendar uses formatted prices for display
            cheapest_prices_for_calendar_raw = cheapest_one_way_prices_raw # Calendar uses raw prices for highlighting
            # Update all_dates_with_data based on dates with calculated prices
            all_dates_with_data = set()
            try:
                 # The keys of cheapest_prices_for_calendar_formatted are the dates with data
                 all_dates_with_data = {isoparse(d).date() for d in cheapest_prices_for_calendar_formatted.keys()}
            except Exception as e:
                 print(f"Error parsing dates for calendar generation from cheapest_prices_for_calendar_formatted keys: {e}")
                 # all_dates_with_data remains empty set

        # Note: If journeys_by_day_one_way_temp is empty, the outer journeys_for_detailed_view,
        # cheapest_prices_for_calendar_formatted, cheapest_prices_for_calendar_raw,
        # and all_dates_with_data remain empty from initialization.


    # --- Generate Calendar Data ---
    # Determine the source of dates for calendar generation based on the mode
    calendar_date_source_temp = {}
    if selected_round_trip_mode:
        # In RT mode, base calendar dates on the outbound journeys fetched (which are in journeys_by_day_outbound_temp)
        calendar_date_source_temp = journeys_by_day_outbound_temp # *** Use the correct source ***
    else:
        # In One-Way mode, base calendar dates on the one-way journeys fetched
        calendar_date_source_temp = journeys_by_day_one_way_temp # *** Use the correct source ***


    # Only generate calendar if there are dates in the source
    if calendar_date_source_temp: # *** Check the selected source ***
        try:
             # The keys of the calendar_date_source_temp are the dates that might have data
             all_dates_with_data = {isoparse(d).date() for d in calendar_date_source_temp.keys()} # *** Use the correct source ***
        except Exception as e:
             print(f"Error parsing dates for calendar generation from source keys: {e}")
             all_dates_with_data = set() # Reset if error





    if all_dates_with_data:
        min_date_obj = min(all_dates_with_data)
        max_date_obj = max(all_dates_with_data)
        current_calendar_month = min_date_obj.replace(day=1)
        end_calendar_month = max_date_obj.replace(day=1)

        while current_calendar_month <= end_calendar_month:
            year = current_calendar_month.year
            month = current_calendar_month.month
            month_name = current_calendar_month.strftime('%B')
            month_weeks = calendar.monthcalendar(year, month)

            weeks_data = []
            for week in month_weeks:
                day_cells_data = []
                for day_num in week:
                    # Initialize with default values
                    day_data = {
                        'day': None,
                        'date_key': None,
                        'price': None, # Formatted price for display
                        'raw_amount': None, # Raw numeric amount for comparison
                        'has_data': False
                    }
                    if day_num != 0:
                        current_day_date_obj = date(year, month, day_num)
                        current_day_date_key = current_day_date_obj.isoformat()

                        day_data['day'] = day_num
                        day_data['date_key'] = current_day_date_key

                        # Check if this date has an entry in the calendar prices
                        # Use the consolidated formatted and raw prices dicts
                        if current_day_date_key in cheapest_prices_for_calendar_formatted: # *** Use the consolidated dict ***
                             day_data['has_data'] = True # Mark as having data even if price is N/A
                             day_data['price'] = cheapest_prices_for_calendar_formatted[current_day_date_key]
                             day_data['raw_amount'] = cheapest_prices_for_calendar_raw[current_day_date_key]


                    day_cells_data.append(day_data)
                weeks_data.append(day_cells_data)

            monthly_calendars.append({
                'month_name': month_name,
                'year': year,
                'weeks': weeks_data
            })
            current_calendar_month = add_months(current_calendar_month, 1)


    # Determine if we are in round trip mode for rendering detailed view
    is_round_trip_detailed_view = selected_round_trip_mode

    return render_template('index.html',
                        exclude_past_journeys=exclude_past_journeys,
                        unique_origins=unique_origins,
                        today_date=date.today().isoformat(),
                        unique_destinations=unique_destinations,
                        unique_processed_dates_str=unique_processed_dates_str,
                        selected_origin=selected_origin,
                        selected_destination=selected_destination,
                        selected_processed_date=selected_processed_date,
                        selected_day_count=selected_day_count,
                        selected_round_trip_mode=selected_round_trip_mode,
                        # Outbound time filter selections (pass original input for form)
                        selected_time_filter_type=selected_time_filter_type,
                        selected_outbound_start_time=original_outbound_start_time,
                        selected_outbound_end_time=original_outbound_end_time,
                        # Return time filter selections (pass original input for form)
                        selected_return_time_filter_type=selected_return_time_filter_type,
                        selected_return_start_time=original_return_start_time,
                        selected_return_end_time=original_return_end_time,
                        # Partial Fare filter selection
                        exclude_partial_fare=exclude_partial_fare,
                        # Highlight Price selection
                        selected_highlight_price_under=selected_highlight_price_under,
                        # Exclude Duration selection
                        selected_exclude_duration_over=selected_exclude_duration_over,
                        # Remove journeys_by_day from here
                        is_round_trip_detailed_view=is_round_trip_detailed_view,
                        monthly_calendars=monthly_calendars,  # <-- Guaranteed initialized and includes raw_amount
                        calendar_day_names=day_names,  # <-- Guaranteed initialized
                        available_stations=available_stations,  # Pass available stations to the template
                        # Helper functions
                        format_duration=format_duration,
                        format_datetime=format_datetime,
                        format_price=format_price)


# --- Run the App ---
if __name__ == '__main__':
    print(f"Attempting to run Flask app. Database expected at: {DATABASE}")
    if not os.path.exists(DATABASE):
         print(f"Error: Database file not found at {DATABASE}. Please run your original script first.")
         # exit(1) # Uncomment to exit if DB is missing
    app.run(debug=True)