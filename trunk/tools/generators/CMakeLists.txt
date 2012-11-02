# Copy over scripts because needed to generate inputs
file(GLOB Sources *.py)
file(COPY ${Sources} DESTINATION .)
