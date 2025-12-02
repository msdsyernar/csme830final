import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox

st.set_page_config(layout="wide", page_title="City Infrastructure Mapper")

# --- 1. Helper Function: Get City Boundary ---
@st.cache_data(ttl=3600)
def get_city_boundary(city, state):
    """
    Fetches the official boundary polygon for a City/State combo.
    Example query: "East Lansing, Michigan, USA"
    """
    query = f"{city}, {state}, USA"
    try:
        # ox.geocode_to_gdf returns a GeoDataFrame with the boundary
        gdf = ox.geocode_to_gdf(query)
        return gdf
    except Exception as e:
        return None

# --- 2. Helper Function: Add Infrastructure ---
def add_infrastructure(m, polygon, infra_type):
    """
    Searches for specific amenities INSIDE the city polygon.
    """
    tags = {}
    color = "blue"
    icon = "info-sign"
    
    # Define what to search for based on user selection
    if infra_type == "Schools":
        tags = {'amenity': ['school', 'university', 'college']}
        color = "blue"
        icon = "book"
    elif infra_type == "Hospitals":
        tags = {'amenity': ['hospital', 'clinic', 'doctors']}
        color = "red"
        icon = "heart"
    elif infra_type == "Parks":
        tags = {'leisure': 'park', 'landuse': 'recreation_ground'}
        color = "green"
        icon = "tree-conifer"
    elif infra_type == "Food":
        tags = {'amenity': ['restaurant', 'cafe', 'fast_food']}
        color = "orange"
        icon = "cutlery"

    try:
        with st.spinner(f"🔍 Searching for {infra_type} in {city}..."):
            # This is the magic line: fetch features strictly inside the polygon
            gdf_infra = ox.features.features_from_polygon(polygon, tags)
            
            if not gdf_infra.empty:
                # Limit to 100 markers to prevent crashing the browser on large cities
                if len(gdf_infra) > 100:
                    st.toast(f"Found {len(gdf_infra)} {infra_type}. Showing top 100.", icon="⚠️")
                    gdf_infra = gdf_infra.head(100)
                else:
                    st.toast(f"Found {len(gdf_infra)} {infra_type}!", icon="✅")

                # Add markers
                # We use the 'centroid' because some results are shapes (buildings), some are points
                for _, row in gdf_infra.iterrows():
                    # Calculate center point for the marker
                    if row.geometry.geom_type == 'Point':
                        loc = [row.geometry.y, row.geometry.x]
                    else:
                        loc = [row.geometry.centroid.y, row.geometry.centroid.x]
                    
                    name = row.get('name', 'Unknown')
                    
                    folium.Marker(
                        location=loc,
                        popup=f"<b>{name}</b><br>Type: {infra_type}",
                        icon=folium.Icon(color=color, icon=icon, prefix='glyphicon'),
                        tooltip=name
                    ).add_to(m)
            else:
                st.warning(f"No {infra_type} found in this city boundaries.")
                
    except Exception as e:
        st.error(f"Error loading {infra_type}: {e}")

# --- 3. Main Application Layout ---
st.title("🏙️ City Infrastructure Visualizer")

with st.sidebar:
    st.header("📍 Location Settings")
    
    # Default values set to East Lansing
    city = st.text_input("City", "East Lansing")
    state = st.text_input("State", "Michigan")
    
    st.divider()
    
    st.header("🏗️ Infrastructure")
    show_boundary = st.toggle("Show City Boundary", value=True)
    
    # Checkboxes for layers
    show_schools = st.checkbox("Schools & Universities")
    show_hospitals = st.checkbox("Hospitals & Clinics")
    show_parks = st.checkbox("Parks & Recreation")
    show_food = st.checkbox("Restaurants (Limit 100)")
    
    st.divider()
    btn = st.button("Update Map", type="primary", use_container_width=True)

# --- 4. Logic to Render Map ---
if btn or city:
    # A. Get the City Boundary
    boundary_gdf = get_city_boundary(city, state)
    
    if boundary_gdf is not None:
        # Extract the polygon geometry
        city_polygon = boundary_gdf.geometry.iloc[0]
        
        # Calculate center of the city for the map initialization
        centroid = city_polygon.centroid
        start_loc = [centroid.y, centroid.x]
        
        # Initialize Folium Map
        m = folium.Map(location=start_loc, zoom_start=13, tiles="CartoDB positron")
        
        # B. Draw the Boundary (if toggled)
        if show_boundary:
            folium.GeoJson(
                boundary_gdf,
                name="City Boundary",
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.1,
                },
                tooltip=f"{city} Boundary"
            ).add_to(m)
            
            # Zoom map to fit the city perfectly
            m.fit_bounds(m.get_bounds())

        # C. Add Infrastructure Layers
        if show_schools: add_infrastructure(m, city_polygon, "Schools")
        if show_hospitals: add_infrastructure(m, city_polygon, "Hospitals")
        if show_parks: add_infrastructure(m, city_polygon, "Parks")
        if show_food: add_infrastructure(m, city_polygon, "Food")

        # D. Display the Map
        st_folium(m, width="100%", height=700)
        
        # Show some stats below the map
        st.info(f"Displaying data for **{city}, {state}**. Data provided by OpenStreetMap.")
        
    else:
        st.error(f"❌ Could not find boundaries for **{city}, {state}**. Please check the spelling.")