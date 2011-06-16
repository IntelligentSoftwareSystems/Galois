# Copy over script because its needed to generate inputs
file(GLOB Sources *.py)
file(COPY ${Sources} DESTINATION .)
