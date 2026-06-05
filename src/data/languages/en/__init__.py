"""English language plugin.

Consolidates transliteration (to Russian / Turkish), city description
template, and query templates from the former ``src/data/english/`` package.
"""
from .. import register_language


class EnglishLanguage:
    """English language plugin implementing the LanguagePlugin protocol."""

    code: str = "en"
    name: str = "English"

    # -- Query templates ------------------------------------------------------

    def templates_single_city(self, city: str) -> list[str]:
        """Return query variants for a bare city name."""
        return [
            # -- short --
            city,
            f"City {city}",
            f"weather {city}",
            f"map of {city}",
            f"{city} population",
            f"history of {city}",
            f"photos {city}",
            f"{city} news",
            f"events in {city}",
            f"jobs in {city}",
            # -- medium --
            f"parks and green spaces in {city}",
            f"free museums and galleries in {city}",
            f"theaters and concert halls in {city}",
            f"street art and graffiti spots in {city}",
            f"flea markets and antique shops in {city}",
            f"walking tours and interesting routes in {city}",
            f"explore {city} in one day",
            f"bike rentals and cycling paths in {city}",
            f"swimming pools and water parks in {city}",
            f"ice rinks and ski slopes near {city}",
            f"zoo and botanical garden in {city}",
            f"escape rooms and entertainment in {city}",
            f"playgrounds and amusement parks in {city}",
            f"temples and religious landmarks in {city}",
            f"architectural highlights of {city}",
            f"best photo spots in {city}",
            f"air quality and environmental data for {city}",
            f"veterinary clinics in {city}",
            f"24-hour pharmacies and hospitals in {city}",
            f"car washes and tire shops in {city}",
            f"laundromats and dry cleaners near downtown {city}",
            f"hair salons and barbershops in {city}",
            f"travel guide for tourists visiting {city}",
            # -- long --
            f"{city} — I want to find out what the climate is like there and how much"
            f" it costs to rent a place for a week",
            f"{city} attracts many tourists, but I would like to know how locals live"
            f" and which neighborhoods are considered residential",
            f"What are the public transport options in {city}"
            f" and how to get from the airport to the city center",
            f"I need to find a dental clinic with good reviews in {city},"
            f" preferably close to downtown",
            f"I want to organize a birthday party in {city} — suggest some unusual venues"
            f" and places for celebrations",
            f"I am interested in street food and night markets in {city},"
            f" where can I grab a cheap bite late in the evening",
            f"I am planning a jog along the waterfront or in a park, recommend running routes in {city}",
            f"Where in {city} can I drop off items for recycling"
            f" and are there collection points near the center",
            f"I am considering relocating and looking at {city} as one of the options,"
            f" tell me about the standard of living and infrastructure",
            f"What coworking spaces and anti-cafes are in {city} with good internet"
            f" and the option to work late",
            f"I recently heard a lot of good things about {city} and want to go there"
            f" for a couple of days — what should I see first",
            f"Can you get around {city} without a car and how well-developed is"
            f" the public transit — subway, buses, trams",
            f"Where is it better to stay in {city} — downtown is pricier but convenient,"
            f" or should I pick a quieter suburb farther out",
        ]

    def templates_single_city_country(
        self, city: str, country: str
    ) -> list[str]:
        """Return query variants for city + country."""
        return [
            # -- short --
            f"{city}, {country}",
            f"City {city}, {country}",
            f"visa to {city}, {country}",
            f"currency in {country}, {city}",
            f"time zone {city}, {country}",
            f"language spoken in {city}, {country}",
            f"holidays in {country}, {city}",
            f"embassy of {country} near {city}",
            # -- medium --
            f"Where in {country} is {city} located?",
            f"do I need a visa to visit {city}, {country}",
            f"what currency should I bring to {city}, {country}",
            f"what vaccinations are needed before traveling to {city}, {country}",
            f"cultural norms and etiquette in {city}, {country}",
            f"tipping customs and payment rules in {city}, {country}",
            f"dress code and behavior rules in {city}, {country}",
            f"can I drink tap water in {city}, {country}",
            f"power outlet types in {country} and {city}",
            f"emergency phone numbers in {city}, {country}",
            f"SIM card and mobile internet in {city}, {country}",
            f"roaming costs in {country} for visitors to {city}",
            f"photography and filming laws in {country}, {city}",
            f"luggage rules for traveling through {country} to {city}",
            f"how taxis and car sharing work in {city}, {country}",
            f"international schools and kindergartens in {city}, {country}",
            f"expat communities in {city}, {country}",
            f"customs regulations when entering {country} via {city}",
            f"VAT refund for tourists in {country}, shopping in {city}",
            f"popular local social networks and messengers in {city}, {country}",
            f"travel health insurance for a trip to {city}, {country}",
            f"rules for importing medication into {country}, city {city}",
            # -- long --
            f"{city}, {country} — what documents do I need to enter"
            f" and is there an e-visa option",
            f"{city} in {country} attracts tourists from around the world,"
            f" but I would like to know how safe it is for solo travelers",
            f"I am planning a business trip to {country} and need to book"
            f" a hotel in {city} for three nights with an airport transfer",
            f"What time zone is {country} in and what time do shops"
            f" and banks open in {city}",
            f"I want to apply for a digital nomad visa to work remotely"
            f" from {city}, {country} — what are the conditions and processing times",
            f"What are the restrictions on bringing alcohol and tobacco"
            f" into {country} when arriving in {city}",
            f"I am planning an extended stay in {country} and want to open"
            f" a bank account in {city} — which banks serve foreigners",
            f"What kind of driver's license do I need in {country} and can I"
            f" rent a car in {city} with my current license",
            f"I heard that {country} has strict drone regulations —"
            f" can I fly a quadcopter in {city}",
            f"I am interested in volunteer programs in {country},"
            f" is there anything suitable in {city} for the summer",
            f"a travel guide to {city} — useful tips before your trip"
            f" and getting to know {country}",
        ]

    def templates_single_city_state(
        self, city: str, state: str
    ) -> list[str]:
        """Return query variants for city + state/region."""
        return [
            # -- short --
            f"{city}, {state}",
            f"City {city}, {state}",
            f"hotels {city}, {state}",
            f"weather {city}, {state}",
            f"zip code {city}, {state}",
            f"demographics {city}, {state}",
            f"crime rate {city}, {state}",
            f"schools in {city}, {state}",
            # -- medium --
            f"how to get to {city} in {state}",
            f"bus schedule and airport transfer in {city}, {state}",
            f"average rent for a one-bedroom in {city}, {state}",
            f"relocating to {state} — considering {city} as an option",
            f"restaurants and cafes near downtown {city}, {state}",
            f"is {city} in {state} safe for families",
            f"universities and colleges in {city}, {state}",
            f"things to do on a weekend in {city}, {state}",
            f"tell me about life in {city}, {state}",
            f"job market and hiring trends in {city}, {state}",
            f"gyms and sports facilities in {city}, {state}",
            f"best bakeries and coffee shops in {city}, {state}",
            f"libraries and community centers in {city}, {state}",
            f"farmers markets and local produce in {city}, {state}",
            f"hiking trails and nature parks near {city}, {state}",
            f"pet-friendly parks and vet clinics in {city}, {state}",
            f"public transport coverage in {city}, {state}",
            f"property taxes and homeowner costs in {city}, {state}",
            f"coworking spaces with fast internet in {city}, {state}",
            f"daycare and preschool options in {city}, {state}",
            f"hospitals and urgent care centers in {city}, {state}",
            f"car dealerships and auto repair shops in {city}, {state}",
            f"annual festivals and community events in {city}, {state}",
            f"recycling and waste collection rules in {city}, {state}",
            # -- long --
            f"I want to find out the average cost of renting an apartment in {city}, {state},"
            f" for a month and what neighborhoods are best",
            f"I am planning to relocate to {state} and considering several"
            f" cities including {city} — what are the pros and cons",
            f"What are the best school districts in {city}, {state},"
            f" for families with young children",
            f"I am looking for a house in {city}, {state}, with a backyard"
            f" and good access to public transport — any recommendations",
            f"How is the startup scene and tech industry doing in {city},"
            f" {state}, and are there many opportunities for developers",
            f"I am thinking of retiring in {city}, {state}, — how is the healthcare,"
            f" cost of living, and overall quality of life there",
            f"What are the commute times from the suburbs of {city}, {state},"
            f" to the downtown area during rush hour",
            f"I need to register a small business in {city}, {state},"
            f" — what permits and licenses are required",
        ]

    def templates_single_city_state_country(
        self, city: str, state: str, country: str
    ) -> list[str]:
        """Return query variants for city + state + country."""
        return [
            # -- short --
            f"{city}, {state}, {country}",
            f"City {city}, {state}, {country}",
            f"zip code {city}, {state}, {country}",
            f"climate {city}, {state}, {country}",
            f"demographics {city}, {state}, {country}",
            f"utilities in {city}, {state}, {country}",
            f"taxes in {city}, {state}, {country}",
            f"police {city}, {state}, {country}",
            # -- medium --
            f"postal code and address format for {city}, {state}, {country}",
            f"where else to go in {country} after visiting {city} in {state}",
            f"what is the climate like in {city}, {state}, {country}",
            f"delivery services and shipping to {city}, {state}, {country}",
            f"consulate information and visa rules for {city}, {state}, {country}",
            f"hotels and hostels in {city}, {state}, {country}",
            f"is {city}, {state}, {country}, worth a visit",
            f"universities and research centers in {city}, {state}, {country}",
            f"things to do in {city}, {state}, {country}",
            f"cost of living breakdown for {city}, {state}, {country}",
            f"local cuisine and must-try restaurants in {city}, {state}, {country}",
            f"how safe is {city} in {state}, {country}, for solo travelers",
            f"public libraries and cultural venues in {city}, {state}, {country}",
            f"best season to visit {city}, {state}, {country}",
            f"internet providers and speed in {city}, {state}, {country}",
            f"local transport options in {city}, {state}, {country}",
            f"real estate market trends in {city}, {state}, {country}",
            f"volunteering and charity work in {city}, {state}, {country}",
            f"import and export regulations for {city}, {state}, {country}",
            f"religious institutions and places of worship in {city}, {state}, {country}",
            f"startup incubators and tech hubs in {city}, {state}, {country}",
            f"water quality and environmental report for {city}, {state}, {country}",
            f"pet adoption and animal shelters in {city}, {state}, {country}",
            # -- long --
            f"I need to send a package to {country}, {state}, to the city of {city},"
            f" what is the postal code and expected delivery time",
            f"How much are the average utility bills for a two-bedroom apartment"
            f" in winter in {city}, {state}, {country}",
            f"Which real estate agencies can best help find housing"
            f" when relocating to {city}, {state}, {country}",
            f"I am considering {city} in {state}, {country}, as a potential"
            f" location for our company's regional office",
            f"What internship and apprenticeship programs are available"
            f" for students in {city}, {state}, {country}",
            f"I want to understand the healthcare system in {city}, {state},"
            f" {country} — how do clinics and insurance work for expats",
            f"What are the rules for setting up a freelance business"
            f" in {city}, {state}, {country}, and what taxes apply",
            f"I am thinking about opening a small café or restaurant"
            f" in {city}, {state}, {country} — what permits are required",
            f"explore the region of {state} in {country} —"
            f" what you need to know when moving to {city}",
        ]

    # -- Two-city query templates ---------------------------------------------

    def templates_two_cities(self, city1: str, city2: str) -> list[str]:
        """Return query variants mentioning two cities."""
        return [
            # -- short --
            f"{city1} and {city2}",
            f"{city1} — {city2}",
            f"{city1} vs {city2}",
            f"{city1} or {city2}",
            f"flights {city1} {city2}",
            f"distance {city1} {city2}",
            f"bus {city1} to {city2}",
            f"train {city1} {city2}",
            # -- medium --
            f"how to get from {city1} to {city2} by train or bus",
            f"which city has cheaper rent, {city1} or {city2}",
            f"weather comparison {city1} vs {city2}",
            f"{city1} or {city2} — which is better for a family vacation",
            f"restaurants and food scene in {city1} compared to {city2}",
            f"moving from {city1} to {city2} — what should I know",
            f"safety comparison between {city1} and {city2}",
            f"nightlife and entertainment in {city1} vs {city2}",
            f"universities in {city1} compared to {city2}",
            f"healthcare quality in {city1} versus {city2}",
            f"cost of groceries in {city1} and {city2}",
            f"internet speed comparison {city1} vs {city2}",
            f"air quality index {city1} compared to {city2}",
            f"pet-friendly places in {city1} and {city2}",
            f"coworking spaces in {city1} vs {city2}",
            f"startup ecosystem in {city1} and {city2}",
            f"commute times in {city1} compared to {city2}",
            f"cultural attractions in {city1} and {city2}",
            f"average salaries in {city1} versus {city2}",
            f"expat communities in {city1} and {city2}",
            f"public transit coverage in {city1} vs {city2}",
            f"green spaces and parks in {city1} and {city2}",
            # -- long --
            f"How long is the flight from {city1} to {city2}"
            f" and which airlines operate this route",
            f"I got job offers in both {city1} and {city2} —"
            f" which city offers a better purchasing power for the salary",
            f"I want to compare the public transport systems of {city1}"
            f" and {city2} — where is it easier to live without a car",
            f"I am interested in the air quality and environmental conditions"
            f" in {city1} and {city2} — which is cleaner",
            f"My family is debating between {city1} and {city2}"
            f" — where are the schools better and is it safer for kids",
            f"If I drive from {city1} to {city2}, what are the must-see stops"
            f" along the way and are there interesting places near the highway",
            f"How do the real estate markets in {city1} and {city2} compare"
            f" — where is it a better time to buy property",
            f"I am a freelancer choosing between {city1} and {city2}"
            f" — where is the coworking scene better and the cost of living lower",
            f"Shipping a parcel between {city1} and {city2}"
            f" — which delivery services operate between these cities",
            f"I am planning a weekend getaway — is it better to explore"
            f" {city1} or {city2} for a short two-day trip",
        ]

    def templates_two_cities_country(
        self, city1: str, country1: str, city2: str, country2: str
    ) -> list[str]:
        """Return query variants for two cities, each with country."""
        return [
            # -- short --
            f"{city1} ({country1}) and {city2} ({country2})",
            f"{city1}, {country1} — {city2}, {country2}",
            f"visa {city1}, {country1}, and {city2}, {country2}",
            f"currency {country1} and {country2}, {city1} and {city2}",
            f"flights {city1} {country1} — {city2} {country2}",
            f"flight {city1} — {city2}, {country1} — {country2}",
            f"climate {city1} ({country1}) and {city2} ({country2})",
            f"prices {city1} {country1} vs {city2} {country2}",
            # -- medium --
            f"what documents do I need to travel from {city1}, {country1}, to {city2}, {country2}",
            f"compare the cost of living in {city1}, {country1}, and {city2}, {country2}",
            f"which is warmer in winter, {city1}, {country1}, or {city2}, {country2}",
            f"quality of life in {city1}, {country1}, vs {city2}, {country2}",
            f"internet and mobile coverage in {city1} ({country1}) and {city2} ({country2})",
            f"customs rules between {country1} and {country2}, {city1} — {city2}",
            f"tax differences between {country1} ({city1}) and {country2} ({city2})",
            f"driving rules in {country1} ({city1}) compared to {country2} ({city2})",
            f"healthcare systems in {city1}, {country1}, versus {city2}, {country2}",
            f"work visa requirements for {country1} ({city1}) and {country2} ({city2})",
            f"digital nomad visas in {country1} and {country2} — {city1} vs {city2}",
            f"safety comparison between {city1} in {country1} and {city2} in {country2}",
            f"time zone difference between {city1}, {country1}, and {city2}, {country2}",
            f"average salaries for IT professionals in {city1}, {country1}, and {city2}, {country2}",
            f"international schools in {city1} ({country1}) vs {city2} ({country2})",
            f"cost of eating out in {city1}, {country1}, compared to {city2}, {country2}",
            f"startup ecosystems in {city1} ({country1}) and {city2} ({country2})",
            f"expat communities and networking in {city1}, {country1}, and {city2}, {country2}",
            f"real estate prices in {city1}, {country1}, versus {city2}, {country2}",
            f"public transportation in {city1} ({country1}) vs {city2} ({country2})",
            f"language barriers for foreigners in {city1}, {country1}, and {city2}, {country2}",
            # -- long --
            f"I want to visit {city1} in {country1} and {city2} in {country2}"
            f" in one trip — how to plan the route and do I need separate visas",
            f"I got job offers in both {city1} ({country1}) and {city2} ({country2}) —"
            f" which city has a higher purchasing power for the same salary",
            f"I want to compare the public transit systems of {city1}, {country1},"
            f" and {city2}, {country2} — where is it easier to live without a car",
            f"I am interested in the air quality and environmental conditions"
            f" in {city1}, {country1}, and {city2}, {country2}",
            f"Which city is better for a family with two kids —"
            f" {city1}, {country1}, or {city2}, {country2}",
            f"I want to understand the difference in purchasing power between {city1}"
            f" ({country1}) and {city2} ({country2}) on the same salary",
            f"I am considering {city1} ({country1}) and {city2} ({country2})"
            f" as options for opening a branch office",
            f"I am going on an internship — choosing between {city1} in {country1}"
            f" and {city2} in {country2}, where are the conditions better for young professionals",
            f"I dream of visiting {country1} and {country2}"
            f" — should I start with {city1} or {city2}",
            f"I want to compare the cost of groceries and daily expenses"
            f" in {city1}, {country1}, versus {city2}, {country2}",
            f"What are the environmental policies and green initiatives"
            f" in {city1} ({country1}) compared to {city2} ({country2})",
        ]

    def templates_two_cities_state(
        self, city1: str, state1: str, city2: str, state2: str
    ) -> list[str]:
        """Return query variants for two cities, each with state."""
        return [
            # -- short --
            f"{city1} ({state1}) and {city2} ({state2})",
            f"{city1}, {state1} — {city2}, {state2}",
            f"distance {city1} {state1} to {city2} {state2}",
            f"rent {city1} ({state1}) vs {city2} ({state2})",
            f"climate {city1} {state1} and {city2} {state2}",
            f"jobs {city1}, {state1}, vs {city2}, {state2}",
            f"schools {city1} ({state1}) {city2} ({state2})",
            f"crime {city1} ({state1}) vs {city2} ({state2})",
            # -- medium --
            f"how to get from {city1}, {state1}, to {city2}, {state2}, by public transport",
            f"compare the cost of living in {city1}, {state1}, and {city2}, {state2}",
            f"I am choosing between {city1} in {state1} and {city2} in {state2} for relocation",
            f"weather comparison between {city1}, {state1}, and {city2}, {state2}",
            f"schools and education in {city1} ({state1}) vs {city2} ({state2})",
            f"job market in {city1}, {state1}, compared to {city2}, {state2}",
            f"property taxes in {city1}, {state1}, versus {city2}, {state2}",
            f"healthcare facilities in {city1} ({state1}) and {city2} ({state2})",
            f"nightlife and dining in {city1}, {state1}, vs {city2}, {state2}",
            f"startup scene in {city1} ({state1}) compared to {city2} ({state2})",
            f"dog-friendly parks in {city1}, {state1}, and {city2}, {state2}",
            f"commute times and traffic in {city1} ({state1}) vs {city2} ({state2})",
            f"gym memberships and fitness options in {city1}, {state1}, and {city2}, {state2}",
            f"local festivals and events in {city1} ({state1}) and {city2} ({state2})",
            f"grocery prices in {city1}, {state1}, compared to {city2}, {state2}",
            f"coworking spaces in {city1} ({state1}) vs {city2} ({state2})",
            f"average salaries in {city1}, {state1}, versus {city2}, {state2}",
            f"museums and cultural venues in {city1} ({state1}) and {city2} ({state2})",
            f"public library systems in {city1}, {state1}, vs {city2}, {state2}",
            f"retirement living in {city1} ({state1}) compared to {city2} ({state2})",
            f"volunteer opportunities in {city1}, {state1}, and {city2}, {state2}",
            # -- long --
            f"I am looking for affordable housing and comparing the real estate markets"
            f" of {city1}, {state1}, and {city2}, {state2}",
            f"Which is a better city for raising kids — {city1} in {state1}"
            f" or {city2} in {state2} — school quality, safety, parks",
            f"My wife and I are considering two places to live — {city1} in {state1}"
            f" or {city2} in {state2}, where is it quieter and cleaner",
            f"I am interested in the IT industry development in {city1}, {state1},"
            f" compared to {city2}, {state2} — where are there more vacancies",
            f"Road trip through {state1} from {city1} to {city2} in {state2}"
            f" — what attractions are worth seeing along the way",
            f"I am comparing internship opportunities for college students"
            f" in {city1} ({state1}) and {city2} ({state2})",
            f"Looking for a summer cottage halfway between {city1} ({state1})"
            f" and {city2} ({state2}) — somewhere easy to reach from both",
            f"Which city is calmer for retirement —"
            f" {city1} in {state1} or {city2} in {state2}, considering healthcare and ecology",
            f"How do the cost of childcare and preschool compare"
            f" between {city1}, {state1}, and {city2}, {state2}",
            f"I am a remote worker deciding between {city1} ({state1})"
            f" and {city2} ({state2}) — internet, cafes, and community",
            f"Thinking of opening a franchise — is it better to start"
            f" in {city1}, {state1}, or {city2}, {state2}",
        ]

    def templates_two_cities_state_country(
        self, city1: str, state1: str, country1: str,
        city2: str, state2: str, country2: str,
    ) -> list[str]:
        """Return query variants for two cities, each with state + country."""
        return [
            # -- short --
            f"{city1}, {state1}, {country1} — {city2}, {state2}, {country2}",
            f"flight {city1} ({state1}, {country1}) — {city2} ({state2}, {country2})",
            f"climate {city1}, {state1}, {country1}, and {city2}, {state2}, {country2}",
            f"prices {city1} ({country1}) vs {city2} ({country2})",
            f"life {city1}, {country1}, and {city2}, {country2}",
            f"salaries {city1} ({country1}) and {city2} ({country2})",
            f"visa {city1} {country1} and {city2} {country2}",
            f"rent {city1} ({state1}) vs {city2} ({state2})",
            # -- medium --
            f"how to get from {city1}, {state1}, {country1}, to {city2}, {state2}, {country2}",
            f"compare the climate in {city1}, {state1}, {country1}, and {city2}, {state2}, {country2}",
            f"flights from {city1} ({state1}, {country1}) to {city2} ({state2}, {country2})",
            f"cost of living in {city1}, {state1}, {country1}, vs {city2}, {state2}, {country2}",
            f"which is better for remote workers, {city1} ({state1}, {country1}) or {city2} ({state2}, {country2})",
            f"tax differences between {city1}, {state1}, {country1}, and {city2}, {state2}, {country2}",
            f"healthcare comparison {city1} ({state1}, {country1}) vs {city2} ({state2}, {country2})",
            f"safety and crime rates in {city1}, {state1}, {country1}, and {city2}, {state2}, {country2}",
            f"school systems in {city1} ({state1}, {country1}) compared to {city2} ({state2}, {country2})",
            f"real estate trends in {city1}, {state1}, {country1}, vs {city2}, {state2}, {country2}",
            f"cultural scene in {city1} ({country1}) and {city2} ({country2})",
            f"internet connectivity in {city1}, {state1}, {country1}, vs {city2}, {state2}, {country2}",
            f"expat blogs about {city1} ({state1}, {country1}) and {city2} ({state2}, {country2})",
            f"weather year-round in {city1}, {country1}, vs {city2}, {country2}",
            f"banking options for foreigners in {city1} ({country1}) and {city2} ({country2})",
            f"apartment rental process in {city1}, {state1}, {country1}, vs {city2}, {state2}, {country2}",
            f"driving license requirements {city1} ({country1}) and {city2} ({country2})",
            f"English proficiency levels in {city1} ({country1}) vs {city2} ({country2})",
            f"delivery services and e-commerce in {city1}, {country1}, and {city2}, {country2}",
            f"nightlife and entertainment in {city1} ({state1}) vs {city2} ({state2})",
            # -- long --
            f"I am moving abroad and comparing {city1} in {state1}, {country1},"
            f" with {city2} in {state2}, {country2} — where is it easier to adapt",
            f"Which city is better for a family with two kids —"
            f" {city1}, {state1}, {country1}, or {city2}, {state2}, {country2}",
            f"I want to understand the difference in purchasing power between {city1}"
            f" ({state1}, {country1}) and {city2} ({state2}, {country2}) on the same salary",
            f"I am considering {city1} ({state1}, {country1}) and {city2} ({state2}, {country2})"
            f" as options for opening a branch office",
            f"I am collecting expat reviews about life in {city1}, {state1}, {country1},"
            f" and {city2}, {state2}, {country2} — sharing experiences on my blog",
            f"I am planning a route — first {city1} ({state1}, {country1}),"
            f" then {city2} ({state2}, {country2}), suggest the best transport option",
            f"Compare the freelancer tax burden in {city1} ({state1}, {country1})"
            f" and {city2} ({state2}, {country2}) — where is it cheaper to register a business",
            f"Visa from {country1} for residents of {country2} — is it easier through"
            f" {city1} ({state1}) or {city2} ({state2})",
            f"I want to compare the nightlife, dating scene, and social life"
            f" in {city1}, {country1}, versus {city2}, {country2}",
            f"What does the job market look like for software engineers"
            f" in {city1} ({state1}, {country1}) compared to {city2} ({state2}, {country2})",
            f"Where is it more convenient to open a bank account as an expat —"
            f" {city1} ({state1}, {country1}) or {city2} ({state2}, {country2})",
            f"I am looking at pet adoption policies and animal shelters"
            f" in {city1}, {state1}, {country1}, vs {city2}, {state2}, {country2}",
        ]

    def templates_two_cities_one_country(
        self, city1: str, country1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with country, city2 bare."""
        return [
            # -- short --
            f"{city1}, {country1} and {city2}",
            f"{city1}, {country1} — {city2}",
            f"{city1} ({country1}) vs {city2}",
            f"route {city1}, {country1}, — {city2}",
            f"tickets {city1} ({country1}) — {city2}",
            f"distance {city1}, {country1}, to {city2}",
            f"{city1} ({country1}) and {city2} — map",
            f"flights {city1}, {country1}, {city2}",
            # -- medium --
            f"distance from {city1} in {country1} to {city2} and the best way to get there",
            f"I am planning a trip to {country1} and want to visit both {city1} and {city2}",
            f"looking for tickets from {city1}, {country1}, to {city2}",
            f"which city is bigger, {city1} in {country1} or {city2}",
            f"food and nightlife in {city1}, {country1}, compared to {city2}",
            f"cost of living comparison {city1} ({country1}) vs {city2}",
            f"weather difference between {city1}, {country1}, and {city2}",
            f"safety comparison {city1} in {country1} versus {city2}",
            f"real estate market in {city1}, {country1}, compared to {city2}",
            f"expat life in {city1} ({country1}) vs {city2}",
            f"healthcare in {city1}, {country1}, versus {city2}",
            f"schools and education in {city1} ({country1}) compared to {city2}",
            f"public transit in {city1}, {country1}, vs {city2}",
            f"average salaries in {city1} ({country1}) and {city2}",
            f"nightlife options in {city1}, {country1}, vs {city2}",
            f"freelance opportunities in {city1} ({country1}) compared to {city2}",
            f"gyms and fitness centers in {city1}, {country1}, and {city2}",
            f"pet-friendly accommodation in {city1} ({country1}) vs {city2}",
            f"coworking spaces in {city1}, {country1}, compared to {city2}",
            f"weekend getaway ideas from {city1} ({country1}) or {city2}",
            # -- long --
            f"I am a digital nomad and comparing {city1} in {country1} with {city2}"
            f" — where are the better coworking spots and lower costs",
            f"I am choosing between {city1} in {country1} and {city2}"
            f" for opening a small business — which has better tax conditions",
            f"Which city is better for a student — {city1} in {country1} or {city2}:"
            f" dormitory costs, part-time jobs, entertainment",
            f"I am looking at real estate and comparing the market in {city1}"
            f" in {country1} with {city2} — where is it smarter to invest in an apartment",
            f"Tours from {country1} to {city2} — is it more convenient"
            f" to go through {city1} or take a direct route",
            f"How does the tech job market in {city1}, {country1},"
            f" compare to {city2} for software engineers",
            f"I am retired and considering either {city1} in {country1} or {city2}"
            f" — where is healthcare better and the cost of living lower",
            f"Planning a road trip from {city1}, {country1}, to {city2}"
            f" — what are the border crossing requirements and scenic stops",
            f"What is the difference in tax rates and business regulations"
            f" between {city1}, {country1}, and {city2}",
            f"I need to ship furniture from {city1} in {country1} to {city2}"
            f" — which international movers are reliable",
            f"How do the dating and social scenes compare between"
            f" {city1} in {country1} and {city2} for young professionals",
            f"I want to learn the local language — which city has better"
            f" language schools, {city1} ({country1}) or {city2}",
        ]

    def templates_two_cities_one_state(
        self, city1: str, state1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with state, city2 bare."""
        return [
            # -- short --
            f"{city1}, {state1} and {city2}",
            f"{city1}, {state1} — {city2}",
            f"{city1} ({state1}) vs {city2}",
            f"road {city1} ({state1}) — {city2}",
            f"train {city1}, {state1}, — {city2}",
            f"distance {city1} ({state1}) to {city2}",
            f"{city1} ({state1}) and {city2} map",
            f"bus {city1} {state1} {city2}",
            # -- medium --
            f"distance from {city1} in {state1} to {city2} and how long it takes to drive",
            f"I want to visit {city1} in {state1} and also stop by {city2} on the way",
            f"are there any buses from {city1} ({state1}) to {city2}",
            f"which has better schools, {city1} in {state1} or {city2}",
            f"real estate prices in {city1}, {state1}, versus {city2}",
            f"weather difference between {city1} ({state1}) and {city2}",
            f"cost of living in {city1}, {state1}, compared to {city2}",
            f"safety comparison {city1} in {state1} versus {city2}",
            f"healthcare options in {city1} ({state1}) and {city2}",
            f"nightlife and dining in {city1}, {state1}, vs {city2}",
            f"average rent in {city1} ({state1}) compared to {city2}",
            f"job market in {city1}, {state1}, versus {city2}",
            f"parks and recreation in {city1} ({state1}) and {city2}",
            f"grocery prices in {city1}, {state1}, versus {city2}",
            f"commute times in {city1} ({state1}) compared to {city2}",
            f"pet-friendly housing in {city1}, {state1}, and {city2}",
            f"internet speed in {city1} ({state1}) vs {city2}",
            f"senior living options in {city1}, {state1}, and {city2}",
            f"daycare and preschool in {city1} ({state1}) compared to {city2}",
            f"volunteer opportunities in {city1}, {state1}, and {city2}",
            # -- long --
            f"I am choosing between {city1} in {state1} and {city2}"
            f" — where is the housing market more affordable for first-time buyers",
            f"Which city is better for families with school-age kids —"
            f" {city1} in {state1} or {city2}, considering schools and safety",
            f"I got a job offer in {city2} but currently live in {city1}, {state1}"
            f" — is the move worth it in terms of salary and quality of life",
            f"My kids are applying to college — compare opportunities for students"
            f" in {city1} ({state1}) and {city2}",
            f"I am looking for a country house halfway between {city1} ({state1})"
            f" and {city2}, easy to reach from both cities",
            f"Which city is calmer for retirement —"
            f" {city1} in {state1} or {city2}, considering healthcare and environment",
            f"Compare the ecology and noise levels in {city1} ({state1}) and {city2}"
            f" — I have a small child and a quiet environment matters",
            f"A tour of the cities in {state1}: {city1} and {city2} — what can you fit"
            f" into a single weekend trip",
            f"I am a remote worker and comparing {city1} ({state1}) with {city2}"
            f" — where is the internet better and the coffee shop scene more vibrant",
            f"How do property taxes compare between {city1} in {state1}"
            f" and {city2} — planning to buy a house soon",
            f"I am comparing water quality and environmental reports"
            f" for {city1} ({state1}) and {city2} — which is healthier",
            f"Which city has more affordable childcare options —"
            f" {city1} in {state1} or {city2}",
        ]

    def templates_two_cities_one_state_country(
        self, city1: str, state1: str, country1: str, city2: str
    ) -> list[str]:
        """Return query variants: city1 with state + country, city2 bare."""
        return [
            # -- short --
            f"{city1}, {state1}, {country1} and {city2}",
            f"{city1}, {state1}, {country1} — {city2}",
            f"flight {city1} ({state1}, {country1}) — {city2}",
            f"tickets {city1}, {country1}, — {city2}",
            f"route {city1} ({country1}) — {city2}",
            f"distance {city1}, {state1}, {country1}, to {city2}",
            f"{city1} ({state1}, {country1}) vs {city2}",
            f"map {city1} {country1} {city2}",
            # -- medium --
            f"distance from {city1} ({state1}, {country1}) to {city2} and the best way to get there",
            f"are there direct flights from {city1}, {state1}, {country1}, to {city2}",
            f"planning a route from {city1} in {state1}, {country1}, to {city2} by car",
            f"climate comparison between {city1} ({state1}, {country1}) and {city2}",
            f"is it cheaper to live in {city1}, {state1}, {country1}, or in {city2}",
            f"cost of living {city1}, {state1}, {country1}, vs {city2}",
            f"weather difference {city1} ({state1}, {country1}) and {city2}",
            f"safety rating of {city1} in {state1}, {country1}, compared to {city2}",
            f"healthcare quality in {city1}, {state1}, {country1}, versus {city2}",
            f"job market in {city1} ({state1}, {country1}) vs {city2}",
            f"schools and education in {city1}, {state1}, {country1}, compared to {city2}",
            f"average rent in {city1} ({state1}, {country1}) versus {city2}",
            f"public transit in {city1}, {state1}, {country1}, vs {city2}",
            f"expat life in {city1} ({state1}, {country1}) compared to {city2}",
            f"nightlife and dining in {city1}, {country1}, vs {city2}",
            f"internet speed in {city1} ({state1}, {country1}) versus {city2}",
            f"coworking spaces {city1}, {state1}, {country1}, and {city2}",
            f"pet-friendly housing in {city1} ({country1}) vs {city2}",
            f"real estate investment in {city1}, {state1}, {country1}, compared to {city2}",
            f"freelance tax rules in {city1} ({state1}, {country1}) vs {city2}",
            f"retirement options in {city1}, {country1}, versus {city2}",
            f"gym and fitness scene in {city1} ({state1}, {country1}) vs {city2}",
            # -- long --
            f"I am choosing between {city1} in {state1}, {country1}, and {city2}"
            f" for opening a small business — where are the tax conditions better",
            f"Which city is better for a student — {city1} ({state1}, {country1})"
            f" or {city2}: dormitory costs, part-time jobs, campus life",
            f"I am traveling from {city1}, {state1}, {country1}, and want to stop"
            f" in {city2} for a couple of days — what is worth seeing",
            f"How does the pace of life in {city1} ({state1}, {country1})"
            f" differ from {city2} — tempo, habits, mentality",
            f"I am looking for flights with luggage from {city1}, {state1}, {country1},"
            f" to {city2} next month — which airlines fly this route",
            f"How different is the cost of groceries in {city1} ({state1}, {country1})"
            f" compared to {city2} — I want to plan a monthly budget",
            f"I am doing a job market analysis for IT professionals — comparing"
            f" opportunities in {city1} ({state1}, {country1}) and {city2}",
            f"Is it worth moving from {city1} in {state1}, {country1}, to {city2}"
            f" — weighing career growth, lifestyle, and cost of living",
            f"What volunteer and charity organizations operate"
            f" in {city1} ({state1}, {country1}) and {city2}",
            f"I am comparing the dating app scene and social events"
            f" in {city1}, {state1}, {country1}, versus {city2}",
        ]

    def templates_two_cities_state_and_country(
        self, city1: str, state1: str, city2: str, country2: str
    ) -> list[str]:
        """Return query variants: city1 with state, city2 with country."""
        return [
            # -- short --
            f"{city1} ({state1}) and {city2} ({country2})",
            f"{city1}, {state1} — {city2}, {country2}",
            f"flights {city1} ({state1}) — {city2} ({country2})",
            f"rent {city1} ({state1}) vs {city2} ({country2})",
            f"weather {city1} {state1} and {city2} {country2}",
            f"jobs {city1} ({state1}) vs {city2} ({country2})",
            f"distance {city1} {state1} to {city2} {country2}",
            f"visa {city1} to {city2}, {country2}",
            # -- medium --
            f"how to get from {city1} in {state1} to {city2} in {country2}",
            f"compare the cost of rent in {city1}, {state1}, and {city2}, {country2}",
            f"looking for tickets from {city1} ({state1}) to {city2}, {country2}",
            f"which has better quality of life, {city1} in {state1} or {city2} in {country2}",
            f"job opportunities in {city1} ({state1}) compared to {city2} ({country2})",
            f"climate difference between {city1}, {state1}, and {city2}, {country2}",
            f"safety comparison {city1} in {state1} vs {city2} in {country2}",
            f"healthcare systems in {city1} ({state1}) and {city2} ({country2})",
            f"schools and universities in {city1}, {state1}, vs {city2}, {country2}",
            f"public transit in {city1} ({state1}) compared to {city2} ({country2})",
            f"real estate market {city1}, {state1}, versus {city2}, {country2}",
            f"cost of groceries in {city1} ({state1}) and {city2} ({country2})",
            f"internet speed and connectivity in {city1}, {state1}, vs {city2}, {country2}",
            f"average salaries in {city1} ({state1}) versus {city2} ({country2})",
            f"expat communities in {city1}, {state1}, and {city2}, {country2}",
            f"nightlife and entertainment in {city1} ({state1}) vs {city2} ({country2})",
            f"coworking spaces in {city1}, {state1}, compared to {city2}, {country2}",
            f"pet-friendly accommodation in {city1} ({state1}) and {city2} ({country2})",
            f"startup scene in {city1}, {state1}, versus {city2}, {country2}",
            f"banking and financial services in {city1} ({state1}) vs {city2} ({country2})",
            f"cultural events and festivals in {city1}, {state1}, and {city2}, {country2}",
            f"gym memberships and sports in {city1} ({state1}) vs {city2} ({country2})",
            # -- long --
            f"I am choosing between {city1} in {state1} and {city2} in {country2}"
            f" — where is the housing market more affordable for young professionals",
            f"Which city is better for a family — {city1} ({state1})"
            f" or {city2} ({country2}), considering schools, safety, and parks",
            f"My friends invited me to move to {city2} ({country2}), but I am used"
            f" to {city1}, {state1} — what are the objective pros and cons",
            f"Which airlines fly from {city1}, {state1},"
            f" to {city2}, {country2}, and how often are there ticket sales",
            f"I am doing a job market analysis for tech workers — comparing"
            f" vacancies in {city1} ({state1}) and {city2} ({country2})",
            f"Which city is more comfortable for retirees — {city1} in {state1}"
            f" or {city2} in {country2}, considering healthcare and cost of living",
            f"I admire {city1} in {state1} but want to compare it with cities"
            f" in {country2}, especially {city2}",
            f"Driving from {city1}, {state1}, to {city2}, {country2}"
            f" — how long is the trip and what are the scenic stops",
            f"I want to compare the freelance tax burden in {city1} ({state1})"
            f" and {city2} ({country2}) — where is it cheaper to register a business",
            f"Planning a two-week vacation — first {city1} in {state1},"
            f" then {city2} in {country2}, what is the best transport between them",
        ]
    
    def city_description(
        self,
        city: str,
        country: str,
        admin1_name: str,
        other_names: list[str],
    ) -> str:
        """Build a short English description of a city."""
        text = f"City {city}, country {country}, state {admin1_name}"
        if other_names:
            text += f". Also called: {', '.join(other_names)}"
        return text


register_language(EnglishLanguage())
