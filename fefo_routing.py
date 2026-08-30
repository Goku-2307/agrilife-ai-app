import math
from typing import Dict, List, Any, Optional, Tuple

# Comprehensive vendor database distributed geographically across the urban & semi-urban retail network
VENDOR_DATABASE: List[Dict[str, Any]] = [
    {
        "id": "VEND-01",
        "name": "Metro Fresh Supermarket Hub",
        "type": "Supermarket Chain",
        "city": "Chennai Central",
        "lat": 13.0827,
        "lon": 80.2707,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 2500, "Banana": 4000, "Apple": 1500, "Mango": 2000},
        "prices_per_kg": {"Tomato": 42.0, "Banana": 38.0, "Apple": 160.0, "Mango": 95.0},
        "has_cold_storage": True,
        "contact": "+91-98401-22345",
        "manager": "A. Kumar (Procurement Head)",
        "operating_hours": "06:00 - 22:00"
    },
    {
        "id": "VEND-02",
        "name": "Koyambedu Wholesale Agro Mandi",
        "type": "Wholesale Mandi",
        "city": "Koyambedu Market",
        "lat": 13.0694,
        "lon": 80.1948,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 8000, "Banana": 12000, "Apple": 5000, "Mango": 6000},
        "prices_per_kg": {"Tomato": 36.0, "Banana": 32.0, "Apple": 140.0, "Mango": 80.0},
        "has_cold_storage": False,
        "contact": "+91-94440-88123",
        "manager": "S. Murugan (Mandi President)",
        "operating_hours": "03:00 - 18:00"
    },
    {
        "id": "VEND-03",
        "name": "Apex Food Processing & Puree Plant",
        "type": "Food Processing Unit",
        "city": "Sriperumbudur Industrial Zone",
        "lat": 12.9675,
        "lon": 79.9406,
        "accepted_crops": ["Tomato", "Mango", "Banana"],
        "demands_kg": {"Tomato": 15000, "Banana": 8000, "Mango": 10000},
        "prices_per_kg": {"Tomato": 30.0, "Banana": 26.0, "Mango": 70.0},
        "has_cold_storage": True,
        "contact": "+91-97890-55432",
        "manager": "Dr. V. Raman (Plant Director)",
        "operating_hours": "24 Hours (Industrial)"
    },
    {
        "id": "VEND-04",
        "name": "GreenHarvest Organic Retail Chain",
        "type": "Premium Organic Retail",
        "city": "Adyar Premium Outlet",
        "lat": 13.0012,
        "lon": 80.2565,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 1200, "Banana": 1800, "Apple": 2000, "Mango": 1500},
        "prices_per_kg": {"Tomato": 55.0, "Banana": 48.0, "Apple": 190.0, "Mango": 120.0},
        "has_cold_storage": True,
        "contact": "+91-99620-11234",
        "manager": "Ms. Preeti (Quality Inspector)",
        "operating_hours": "07:00 - 21:30"
    },
    {
        "id": "VEND-05",
        "name": "Tambaram Regional Distribution Center",
        "type": "Regional Hub",
        "city": "Tambaram South",
        "lat": 12.9249,
        "lon": 80.1000,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 5000, "Banana": 6000, "Apple": 3000, "Mango": 4000},
        "prices_per_kg": {"Tomato": 38.0, "Banana": 34.0, "Apple": 150.0, "Mango": 85.0},
        "has_cold_storage": True,
        "contact": "+91-98840-77654",
        "manager": "R. Selvam (Logistics Lead)",
        "operating_hours": "05:00 - 23:00"
    },
    {
        "id": "VEND-06",
        "name": "Anna Nagar Mega Hypermarket",
        "type": "Hypermarket Hub",
        "city": "Anna Nagar West",
        "lat": 13.0850,
        "lon": 80.2100,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 3200, "Banana": 4500, "Apple": 2200, "Mango": 2800},
        "prices_per_kg": {"Tomato": 44.0, "Banana": 39.0, "Apple": 165.0, "Mango": 98.0},
        "has_cold_storage": True,
        "contact": "+91-98410-33445",
        "manager": "K. Balaji (Store Operations)",
        "operating_hours": "07:00 - 22:30"
    },
    {
        "id": "VEND-07",
        "name": "Velachery Fresh Agro Express",
        "type": "Express Urban Mart",
        "city": "Velachery Bypass",
        "lat": 12.9815,
        "lon": 80.2180,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 2000, "Banana": 3000, "Apple": 1800, "Mango": 1900},
        "prices_per_kg": {"Tomato": 46.0, "Banana": 40.0, "Apple": 170.0, "Mango": 105.0},
        "has_cold_storage": True,
        "contact": "+91-98402-99881",
        "manager": "M. Suresh (Inventory Lead)",
        "operating_hours": "06:30 - 22:00"
    },
    {
        "id": "VEND-08",
        "name": "Ambattur Industrial Agro Aggregator",
        "type": "B2B Aggregator Depot",
        "city": "Ambattur Industrial Estate",
        "lat": 13.1143,
        "lon": 80.1548,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 7000, "Banana": 9000, "Apple": 4000, "Mango": 5000},
        "prices_per_kg": {"Tomato": 37.0, "Banana": 33.0, "Apple": 145.0, "Mango": 82.0},
        "has_cold_storage": True,
        "contact": "+91-97900-11223",
        "manager": "P. Natarajan (Dispatch Controller)",
        "operating_hours": "05:00 - 21:00"
    },
    {
        "id": "VEND-09",
        "name": "Porur Cold-Chain Logistics Hub",
        "type": "Cold Storage & Depot",
        "city": "Porur Junction",
        "lat": 13.0382,
        "lon": 80.1565,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 4500, "Banana": 5500, "Apple": 3500, "Mango": 3000},
        "prices_per_kg": {"Tomato": 40.0, "Banana": 36.0, "Apple": 155.0, "Mango": 90.0},
        "has_cold_storage": True,
        "contact": "+91-98845-66778",
        "manager": "G. Dinesh (Cold-Chain Supervisor)",
        "operating_hours": "24 Hours (Cold Facility)"
    },
    {
        "id": "VEND-10",
        "name": "OMR Tech Corridor Superstore",
        "type": "Urban Superstore",
        "city": "Sholinganallur OMR",
        "lat": 12.9010,
        "lon": 80.2279,
        "accepted_crops": ["Tomato", "Banana", "Apple", "Mango"],
        "demands_kg": {"Tomato": 2800, "Banana": 3500, "Apple": 2500, "Mango": 2200},
        "prices_per_kg": {"Tomato": 48.0, "Banana": 42.0, "Apple": 175.0, "Mango": 110.0},
        "has_cold_storage": True,
        "contact": "+91-99401-44556",
        "manager": "T. Arvind (Commercial Manager)",
        "operating_hours": "07:00 - 23:00"
    }
]


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two geographic coordinates in kilometers"""
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return round(R * c, 2)


def generate_route_waypoints(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    num_points: int = 12
) -> List[Dict[str, float]]:
    """
    Generates realistic road-corridor waypoints connecting Truck start to Shop destination
    """
    waypoints = []
    # Subtle curvature to simulate realistic road geometry
    mid_lat = (start_lat + end_lat) / 2.0
    mid_lon = (start_lon + end_lon) / 2.0
    curve_offset_lat = (end_lon - start_lon) * 0.08
    curve_offset_lon = -(end_lat - start_lat) * 0.08

    for i in range(num_points + 1):
        t = i / float(num_points)
        # Quadratic Bezier interpolation for smooth road curve
        w_start = (1 - t) ** 2
        w_mid = 2 * (1 - t) * t
        w_end = t ** 2
        lat = w_start * start_lat + w_mid * (mid_lat + curve_offset_lat) + w_end * end_lat
        lon = w_start * start_lon + w_mid * (mid_lon + curve_offset_lon) + w_end * end_lon
        waypoints.append({"lat": round(lat, 5), "lon": round(lon, 5)})

    return waypoints


class FEFORoutingEngine:
    """
    FEFO (First-Expired, First-Out) Multi-Shipment Prioritizer and AI Route Optimizer
    """
    def __init__(self, vendors: Optional[List[Dict[str, Any]]] = None):
        self.vendors = vendors if vendors is not None else VENDOR_DATABASE
        self.average_speed_kmh = 42.0  # Average commercial truck transit speed in km/h
        self.fuel_cost_per_km = 8.5    # Fuel + operational cost per km in INR

    def rank_fefo_queue(self, shipments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts active shipments by remaining shelf life (ascending) -> Least Remaining Shelf Life First
        """
        sorted_shipments = sorted(
            shipments,
            key=lambda s: (s.get("RSL_final_days", 999.0), -s.get("fefo_priority", 1))
        )

        for rank, s in enumerate(sorted_shipments, start=1):
            s["fefo_rank"] = rank
            rsl = s.get("RSL_final_days", 10.0)
            if rsl <= 2.0:
                s["dispatch_urgency"] = "🚨 IMMEDIATE DISPATCH (FEFO Critical)"
                s["urgency_badge"] = "CRITICAL"
            elif rsl <= 4.5:
                s["dispatch_urgency"] = "⚠️ EXPEDITE SHIPMENT (FEFO Priority 2)"
                s["urgency_badge"] = "MODERATE"
            else:
                s["dispatch_urgency"] = "✅ SCHEDULED DISPATCH (FEFO Normal)"
                s["urgency_badge"] = "OPTIMAL"

        return sorted_shipments

    def optimize_vendor_route(
        self,
        crop_name: str,
        shipment_quantity_kg: float,
        rsl_final_days: float,
        truck_lat: float = 13.0400,
        truck_lon: float = 80.1200,
        shipment_id: str = "SH001"
    ) -> Dict[str, Any]:
        """
        Evaluates all nearby shops/mandis/depots and selects the single best destination
        where the truck should go next based on:
        1. Crop match and current demand capacity (kg)
        2. Transit time vs. Remaining Shelf Life safety buffer margin
        3. Commercial net profit (Gross Revenue - Transport Cost)
        4. Urgent spoilage avoidance penalty if RSL is low
        """
        clean_crop = crop_name.capitalize()
        rsl_hours = rsl_final_days * 24.0

        scored_candidates = []

        for v in self.vendors:
            if clean_crop not in v["accepted_crops"]:
                continue

            dist_km = haversine_distance_km(truck_lat, truck_lon, v["lat"], v["lon"])
            # Estimated transit hours with traffic margin
            transit_hours = round(dist_km / self.average_speed_kmh, 2)
            transit_minutes = int(transit_hours * 60)
            margin_hours = round(rsl_hours - transit_hours, 2)

            # Check feasibility: Truck must arrive before shelf life expires
            is_feasible = margin_hours > 3.0  # At least 3 hours buffer on arrival

            price_per_kg = v["prices_per_kg"].get(clean_crop, 30.0)
            demand_kg = v["demands_kg"].get(clean_crop, 5000)
            accepted_qty = min(shipment_quantity_kg, demand_kg)
            
            gross_revenue = accepted_qty * price_per_kg
            transport_cost = dist_km * self.fuel_cost_per_km
            net_profit = gross_revenue - transport_cost

            # Multi-objective optimization score
            if is_feasible:
                # If RSL is critical (< 2 days), heavily weight proximity and cold storage to avoid spoilage
                if rsl_final_days <= 2.0:
                    distance_penalty = dist_km * 60.0
                    cold_bonus = 2000.0 if v["has_cold_storage"] else 0.0
                    score = net_profit - distance_penalty + (margin_hours * 120.0) + cold_bonus
                else:
                    # Profit-maximizing commercial route
                    score = net_profit - (dist_km * 12.0)
            else:
                score = -100000.0  # Infeasible due to imminent spoilage risk

            route_name = f"Route to {v['city']} via {'Expressway Corridor' if dist_km > 18 else 'Urban Arterial'}"

            scored_candidates.append({
                "vendor_id": v["id"],
                "vendor_name": v["name"],
                "type": v.get("type", "Retailer"),
                "city": v["city"],
                "lat": v["lat"],
                "lon": v["lon"],
                "distance_km": dist_km,
                "transit_hours": transit_hours,
                "transit_minutes": transit_minutes,
                "margin_hours": margin_hours,
                "price_per_kg": price_per_kg,
                "demand_kg": demand_kg,
                "accepted_qty_kg": accepted_qty,
                "gross_revenue": round(gross_revenue, 2),
                "transport_cost": round(transport_cost, 2),
                "net_profit": round(net_profit, 2),
                "is_feasible": is_feasible,
                "score": round(score, 2),
                "route_name": route_name,
                "has_cold_storage": v["has_cold_storage"],
                "contact": v["contact"],
                "manager": v.get("manager", "Store Lead"),
                "operating_hours": v.get("operating_hours", "08:00 - 20:00")
            })

        # Sort by optimization score descending
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)

        if not scored_candidates:
            return {
                "recommended_vendor": None,
                "all_candidates": [],
                "route_waypoints": [],
                "rationale": "No suitable shops found matching current cargo requirements."
            }

        best = scored_candidates[0]

        # Generate smooth route waypoints from truck to best vendor
        waypoints = generate_route_waypoints(truck_lat, truck_lon, best["lat"], best["lon"])

        # Generate clear rationale
        if rsl_final_days <= 2.0:
            rationale = (
                f"🚨 **Urgent FEFO Spoilage Protection:** Shipment **{shipment_id}** has limited shelf life "
                f"({rsl_final_days:.1f} days / {rsl_hours:.1f}h remaining). Truck is routed to **{best['vendor_name']}** "
                f"({best['distance_km']} km, ~{best['transit_minutes']} mins transit) with high demand of "
                f"**{best['demand_kg']:,} kg** and safety margin of **{best['margin_hours']}h**."
            )
        else:
            rationale = (
                f"✅ **Optimal Profit Commercial Route:** Shipment **{shipment_id}** has healthy shelf life "
                f"({rsl_final_days:.1f} days). Truck is directed to **{best['vendor_name']}** ({best['type']}) "
                f"offering premium price of **₹{best['price_per_kg']}/kg**, delivering maximum net profit of **₹{best['net_profit']:,}** "
                f"across {best['distance_km']} km (~{best['transit_minutes']} mins)."
            )

        return {
            "recommended_vendor": best,
            "all_candidates": scored_candidates,
            "truck_location": {"lat": truck_lat, "lon": truck_lon},
            "route_waypoints": waypoints,
            "rationale": rationale,
            "recommended_route": best["route_name"],
            "eta_hours": best["transit_hours"],
            "eta_minutes": best["transit_minutes"],
            "safety_margin_hours": best["margin_hours"]
        }
