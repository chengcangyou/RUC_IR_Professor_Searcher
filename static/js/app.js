// Global state
let filterOptions = null;
let searchResults = null;
let currentPage = 1;
const RESULTS_PER_PAGE = 5;

// Region mapping
const REGION_MAPPING = {
    'asia': 'Asia',
    'europe': 'Europe',
    'africa': 'Africa',
    'north-america': 'Northern America',
    'south-america': 'Latin America and the Caribbean',
    'oceania': 'Oceania'
};

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    loadFilterOptions();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Search button
    document.getElementById('searchBtn').addEventListener('click', performSearch);
    
    // Enter key in search input
    document.getElementById('searchInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    // Region toggle
    document.querySelectorAll('.region-header').forEach(header => {
        header.addEventListener('click', function(e) {
            // Don't toggle if clicking on checkbox
            if (e.target.type === 'checkbox') return;
            
            const countryList = this.nextElementSibling;
            const toggleIcon = this.querySelector('.toggle-icon');
            
            if (countryList.style.display === 'none') {
                countryList.style.display = 'block';
                toggleIcon.classList.add('expanded');
            } else {
                countryList.style.display = 'none';
                toggleIcon.classList.remove('expanded');
            }
        });
    });
    
    // Clear filter
    document.getElementById('clearFilter').addEventListener('click', clearFilters);
    
    // Pagination
    document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
    document.getElementById('nextPage').addEventListener('click', () => changePage(1));
}

// Setup region-country linkage after filters are populated
function setupRegionCountryLinkage() {
    // Region checkbox change handler
    document.querySelectorAll('.region-checkbox').forEach(regionCheckbox => {
        regionCheckbox.addEventListener('change', function() {
            const regionHeader = this.closest('.region-header');
            const countryList = regionHeader.nextElementSibling;
            const countryCheckboxes = countryList.querySelectorAll('.country-checkbox');
            
            // When region is checked, check all countries
            // When region is unchecked, uncheck all countries
            countryCheckboxes.forEach(countryCheckbox => {
                countryCheckbox.checked = this.checked;
            });
        });
    });
    
    // Country checkbox change handler
    document.querySelectorAll('.country-checkbox').forEach(countryCheckbox => {
        countryCheckbox.addEventListener('change', function() {
            const countryList = this.closest('.country-list');
            const regionHeader = countryList.previousElementSibling;
            const regionCheckbox = regionHeader.querySelector('.region-checkbox');
            const allCountryCheckboxes = countryList.querySelectorAll('.country-checkbox');
            
            // Check if all countries are checked
            const allChecked = Array.from(allCountryCheckboxes).every(cb => cb.checked);
            // Check if any country is checked
            const anyChecked = Array.from(allCountryCheckboxes).some(cb => cb.checked);
            
            if (allChecked) {
                // If all countries are checked, check the region
                regionCheckbox.checked = true;
            } else if (!anyChecked) {
                // If no countries are checked, uncheck the region
                regionCheckbox.checked = false;
            } else {
                // If some (but not all) countries are checked, uncheck the region
                // This ensures the region checkbox only stays checked when ALL countries are checked
                regionCheckbox.checked = false;
            }
        });
    });
}

// Load filter options
async function loadFilterOptions() {
    try {
        const response = await fetch('/api/filter-options');
        const data = await response.json();
        
        if (data.success) {
            filterOptions = data.data;
            populateFilters();
        } else {
            console.error('Failed to load filter options:', data.error);
        }
    } catch (error) {
        console.error('Error loading filter options:', error);
    }
}

// Populate filter selects
function populateFilters() {
    // Populate countries for each region
    document.querySelectorAll('.region-header').forEach(header => {
        const regionKey = header.dataset.region;
        const regionName = REGION_MAPPING[regionKey];
        const countryList = header.nextElementSibling;
        
        // Filter countries by region/sub-region
        const countries = filterOptions.countries.filter(country => {
            if (regionName === 'Northern America') {
                return country.sub_region === 'Northern America';
            } else if (regionName === 'Latin America and the Caribbean') {
                return country.sub_region === 'Latin America and the Caribbean' || 
                       country.sub_region === 'South America' ||
                       country.sub_region === 'Central America' ||
                       country.sub_region === 'Caribbean';
            } else {
                return country.region === regionName;
            }
        });
        
        // Sort countries by name
        countries.sort((a, b) => a.name.localeCompare(b.name));
        
        // Create country checkboxes
        countryList.innerHTML = '';
        countries.forEach(country => {
            const div = document.createElement('div');
            div.className = 'country-item';
            div.innerHTML = `
                <label>
                    <input type="checkbox" class="country-checkbox" value="${country.alpha_2}">
                    ${country.name}
                </label>
            `;
            countryList.appendChild(div);
        });
    });
    
    // Setup region-country linkage after all filters are populated
    setupRegionCountryLinkage();
}

// Get selected filters
function getSelectedFilters() {
    // Get selected regions and sub-regions
    const selectedRegions = [];
    const selectedSubRegions = [];
    
    document.querySelectorAll('.region-checkbox:checked').forEach(checkbox => {
        const value = checkbox.value;
        // Check if it's a main region or sub-region
        if (value === 'Asia' || value === 'Europe' || value === 'Africa' || value === 'Oceania') {
            selectedRegions.push(value);
        } else if (value === 'Northern America' || value === 'Latin America and the Caribbean') {
            selectedSubRegions.push(value);
        }
    });
    
    // Get selected countries within regions
    const selectedCountries = [];
    document.querySelectorAll('.country-checkbox:checked').forEach(checkbox => {
        selectedCountries.push(checkbox.value);
    });
    
    return {
        regions: selectedRegions,
        sub_regions: selectedSubRegions,
        countries: selectedCountries
    };
}

// Clear filters
function clearFilters() {
    document.querySelectorAll('.region-checkbox, .country-checkbox').forEach(checkbox => {
        checkbox.checked = false;
    });
    
    // Collapse all regions
    document.querySelectorAll('.country-list').forEach(list => {
        list.style.display = 'none';
    });
    document.querySelectorAll('.toggle-icon').forEach(icon => {
        icon.classList.remove('expanded');
    });
}

// Perform search
async function performSearch() {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        alert('请输入搜索关键词');
        return;
    }
    
    // Get filters
    const filters = getSelectedFilters();
    
    // Show loading
    showLoading(true);
    
    // Disable search button
    const searchBtn = document.getElementById('searchBtn');
    searchBtn.disabled = true;
    searchBtn.querySelector('.btn-text').style.display = 'none';
    searchBtn.querySelector('.btn-loading').style.display = 'inline';
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                regions: filters.regions,
                sub_regions: filters.sub_regions,
                countries: filters.countries,
                top_k: 50
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            searchResults = data.data;
            currentPage = 1;
            displayResults();
        } else {
            alert('搜索失败: ' + data.error);
        }
    } catch (error) {
        console.error('Search error:', error);
        alert('搜索出错，请稍后重试');
    } finally {
        showLoading(false);
        searchBtn.disabled = false;
        searchBtn.querySelector('.btn-text').style.display = 'inline';
        searchBtn.querySelector('.btn-loading').style.display = 'none';
    }
}

// Display search results
function displayResults() {
    const { query, results, ai_summary } = searchResults;
    
    // Hide welcome message
    document.getElementById('welcomeMessage').style.display = 'none';
    
    // Display AI summary with Markdown rendering
    if (ai_summary) {
        const summarySection = document.getElementById('summarySection');
        const summaryContent = document.getElementById('summaryContent');
        
        // Render Markdown to HTML
        if (typeof marked !== 'undefined') {
            summaryContent.innerHTML = marked.parse(ai_summary);
        } else {
            summaryContent.textContent = ai_summary;
        }
        
        summarySection.style.display = 'block';
    }
    
    // Display results
    const resultsSection = document.getElementById('resultsSection');
    const resultsCount = document.getElementById('resultsCount');
    
    resultsCount.textContent = `共 ${results.length} 个结果`;
    
    // Calculate pagination
    const totalPages = Math.ceil(results.length / RESULTS_PER_PAGE);
    updatePagination(totalPages);
    
    // Display current page results
    displayPage(currentPage);
    
    resultsSection.style.display = 'block';
    
    // Scroll to results
    setTimeout(() => {
        document.getElementById('summarySection').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Display specific page
function displayPage(page) {
    const { results } = searchResults;
    const resultsContainer = document.getElementById('resultsContainer');
    
    // Calculate start and end indices
    const startIdx = (page - 1) * RESULTS_PER_PAGE;
    const endIdx = Math.min(startIdx + RESULTS_PER_PAGE, results.length);
    
    // Clear container
    resultsContainer.innerHTML = '';
    
    // Display results for current page
    for (let i = startIdx; i < endIdx; i++) {
        const professor = results[i];
        const card = createProfessorCard(professor);
        resultsContainer.appendChild(card);
    }
    
    // Scroll to top of results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update pagination controls
function updatePagination(totalPages) {
    document.getElementById('currentPage').textContent = currentPage;
    document.getElementById('totalPages').textContent = totalPages;
    
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === totalPages;
}

// Change page
function changePage(delta) {
    const totalPages = Math.ceil(searchResults.results.length / RESULTS_PER_PAGE);
    const newPage = currentPage + delta;
    
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        displayPage(currentPage);
        updatePagination(totalPages);
    }
}

// Create professor card
function createProfessorCard(professor) {
    const card = document.createElement('div');
    card.className = 'professor-card';
    
    // Rank badge
    const rank = document.createElement('div');
    rank.className = 'professor-rank';
    rank.textContent = professor.rank;
    card.appendChild(rank);
    
    // Name
    const name = document.createElement('div');
    name.className = 'professor-name';
    name.textContent = professor.name;
    card.appendChild(name);
    
    // Institution and country
    const institution = document.createElement('div');
    institution.className = 'professor-institution';
    institution.innerHTML = `
        🏫 ${professor.institution}
        <span class="professor-country">${professor.country}</span>
    `;
    card.appendChild(institution);
    
    // Similarity score
    const score = document.createElement('div');
    score.className = 'professor-score';
    score.innerHTML = `
        <span class="score-label">相似度:</span>
        <span class="score-value">${(professor.similarity_score * 100).toFixed(2)}%</span>
        <span class="score-label">| 内容长度: ${professor.content_length.toLocaleString()} 字符</span>
    `;
    card.appendChild(score);
    
    // Homepage link
    if (professor.homepage) {
        const homepage = document.createElement('a');
        homepage.className = 'professor-homepage';
        homepage.href = professor.homepage;
        homepage.target = '_blank';
        homepage.textContent = '🔗 访问主页';
        card.appendChild(homepage);
    }
    
    // Snippet
    if (professor.snippet) {
        const snippet = document.createElement('div');
        snippet.className = 'professor-snippet';
        snippet.textContent = professor.snippet;
        card.appendChild(snippet);
    }
    
    return card;
}

// Show/hide loading overlay
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = show ? 'flex' : 'none';
}
