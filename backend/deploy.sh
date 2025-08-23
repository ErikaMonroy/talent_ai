#!/bin/bash

# =============================================================================
# TalentAI Docker Deployment Script
# =============================================================================
# This script helps deploy TalentAI using Docker Compose

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are available"
}

# Create .env file if it doesn't exist
setup_env() {
    if [ ! -f ".env" ]; then
        print_info "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration before running the application"
        print_info "Important: Change the SECRET_KEY and database passwords in production!"
    else
        print_info ".env file already exists"
    fi
}

# Create necessary directories
setup_directories() {
    print_info "Creating necessary directories..."
    mkdir -p backend/logs
    mkdir -p backend/models
    print_success "Directories created"
}

# Build and start services
start_services() {
    print_info "Building and starting TalentAI services..."
    
    # Use docker compose or docker-compose based on availability
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Build and start services
    $COMPOSE_CMD up --build -d
    
    print_success "Services started successfully!"
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    check_services_health
}

# Check services health
check_services_health() {
    print_info "Checking services health..."
    
    # Check PostgreSQL
    if docker ps | grep -q "talentai_postgres.*Up"; then
        print_success "PostgreSQL is running"
    else
        print_error "PostgreSQL is not running properly"
    fi
    
    # Check Backend API
    if docker ps | grep -q "talentai_backend.*Up"; then
        print_success "Backend API is running"
        
        # Test API health endpoint
        sleep 5
        if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
            print_success "Backend API health check passed"
        else
            print_warning "Backend API health check failed - service might still be starting"
        fi
    else
        print_error "Backend API is not running properly"
    fi
    
    # Check Redis
    if docker ps | grep -q "talentai_redis.*Up"; then
        print_success "Redis is running"
    else
        print_warning "Redis is not running"
    fi
}

# Stop services
stop_services() {
    print_info "Stopping TalentAI services..."
    
    if docker compose version &> /dev/null; then
        docker compose down
    else
        docker-compose down
    fi
    
    print_success "Services stopped"
}

# Show logs
show_logs() {
    if docker compose version &> /dev/null; then
        docker compose logs -f
    else
        docker-compose logs -f
    fi
}

# Show status
show_status() {
    print_info "TalentAI Services Status:"
    echo ""
    docker ps --filter "name=talentai" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    
    print_info "Available endpoints:"
    echo "  - API Documentation: http://localhost:8000/docs"
    echo "  - API Health Check: http://localhost:8000/api/v1/health"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Redis: localhost:6379"
}

# Main script
case "$1" in
    "start")
        print_info "Starting TalentAI deployment..."
        check_docker
        setup_env
        setup_directories
        start_services
        show_status
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 2
        start_services
        show_status
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "health")
        check_services_health
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|health}"
        echo ""
        echo "Commands:"
        echo "  start   - Build and start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        echo "  health  - Check service health"
        exit 1
        ;;
esac