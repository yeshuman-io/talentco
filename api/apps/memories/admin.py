from django.contrib import admin
from django.utils.html import format_html
from .models import Memory, MemoryRelation, MemorySearch, MemoryPattern


@admin.register(Memory)
class MemoryAdmin(admin.ModelAdmin):
    list_display = [
        'user_id', 'truncated_content', 'memory_type', 'category', 
        'subcategory', 'importance', 'interaction_type', 'requires_followup', 'created_at'
    ]
    list_filter = [
        'memory_type', 'category', 'subcategory', 'importance', 'interaction_type',
        'requires_followup', 'is_archived', 'source', 'created_at'
    ]
    search_fields = ['user_id', 'content', 'session_id', 'category', 'subcategory']
    readonly_fields = ['id', 'created_at', 'updated_at', 'embedding']
    
    fieldsets = (
        ('Core Information', {
            'fields': ('id', 'user_id', 'content', 'session_id')
        }),
        ('Categorization', {
            'fields': ('memory_type', 'interaction_type', 'category', 'subcategory', 'importance')
        }),
        ('Flags & Status', {
            'fields': ('requires_followup', 'is_archived', 'source')
        }),
        ('Technical', {
            'fields': ('embedding', 'metadata'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def truncated_content(self, obj):
        if len(obj.content) > 50:
            return f"{obj.content[:50]}..."
        return obj.content
    truncated_content.short_description = 'Content'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related()


@admin.register(MemoryRelation)
class MemoryRelationAdmin(admin.ModelAdmin):
    list_display = [
        'from_memory_summary', 'relation_type', 'to_memory_summary', 
        'strength', 'created_at'
    ]
    list_filter = ['relation_type', 'strength', 'created_at']
    search_fields = [
        'from_memory__content', 'to_memory__content', 
        'from_memory__user_id', 'to_memory__user_id'
    ]
    readonly_fields = ['created_at']
    
    def from_memory_summary(self, obj):
        return f"{obj.from_memory.user_id}: {obj.from_memory.content[:30]}..."
    from_memory_summary.short_description = 'From Memory'
    
    def to_memory_summary(self, obj):
        return f"{obj.to_memory.user_id}: {obj.to_memory.content[:30]}..."
    to_memory_summary.short_description = 'To Memory'


@admin.register(MemorySearch)
class MemorySearchAdmin(admin.ModelAdmin):
    list_display = [
        'user_id', 'truncated_query', 'search_type', 
        'results_count', 'search_duration_ms', 'created_at'
    ]
    list_filter = ['search_type', 'results_count', 'created_at']
    search_fields = ['user_id', 'query', 'session_id']
    readonly_fields = ['query_embedding', 'created_at']
    
    def truncated_query(self, obj):
        if len(obj.query) > 40:
            return f"{obj.query[:40]}..."
        return obj.query
    truncated_query.short_description = 'Query'


@admin.register(MemoryPattern)
class MemoryPatternAdmin(admin.ModelAdmin):
    list_display = [
        'user_id', 'title', 'pattern_type', 'category', 'confidence_display',
        'supporting_count', 'is_active', 'last_updated'
    ]
    list_filter = [
        'pattern_type', 'category', 'is_active', 'confidence', 
        'last_updated', 'created_at'
    ]
    search_fields = ['user_id', 'title', 'description', 'category', 'pattern_type']
    readonly_fields = ['created_at', 'last_updated']
    filter_horizontal = ['supporting_memories']
    
    def confidence_display(self, obj):
        color = 'green' if obj.confidence > 0.7 else 'orange' if obj.confidence > 0.4 else 'red'
        return format_html(
            '<span style="color: {};">{:.2f}</span>',
            color, obj.confidence
        )
    confidence_display.short_description = 'Confidence'
    confidence_display.admin_order_field = 'confidence'
    
    def supporting_count(self, obj):
        return obj.supporting_memories.count()
    supporting_count.short_description = '# Supporting Memories' 