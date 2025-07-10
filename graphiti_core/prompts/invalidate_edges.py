"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class InvalidatedEdges(BaseModel):
    contradicted_facts: list[int] = Field(
        ...,
        description='List of ids of facts that should be invalidated. If no facts should be invalidated, the list should be empty.',
    )


class Prompt(Protocol):
    v1: PromptVersion
    v2: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction
    v2: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an AI assistant that helps determine which relationships in a knowledge graph should be invalidated based solely on explicit contradictions in newer information.',
        ),
        Message(
            role='user',
            content=f"""
               Based on the provided existing edges and new edges with their timestamps, determine which relationships, if any, should be marked as expired due to contradictions or updates in the newer edges.
               Use the start and end dates of the edges to determine which edges are to be marked expired.
                Only mark a relationship as invalid if there is clear evidence from other edges that the relationship is no longer true.
                Do not invalidate relationships merely because they weren't mentioned in the episodes. You may use the current episode and previous episodes as well as the facts of each edge to understand the context of the relationships.

                Previous Episodes:
                {context['previous_episodes']}

                Current Episode:
                {context['current_episode']}

                Existing Edges (sorted by timestamp, newest first):
                {context['existing_edges']}

                New Edges:
                {context['new_edges']}

                Each edge is formatted as: "UUID | SOURCE_NODE - EDGE_NAME - TARGET_NODE (fact: EDGE_FACT), START_DATE (END_DATE, optional))"
            """,
        ),
    ]


def v2(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are an AI assistant that identifies contradictions between facts in a knowledge graph, with special attention to information that evolves over time and belongs to specific entities.',
        ),
        Message(
            role='user',
            content=f"""
               Determine which existing facts are contradicted by the new fact. A contradiction occurs when:
               
               1. Facts contain directly conflicting information about the same attribute of the same entity (e.g., different values, status changes)
               2. Facts represent different states of the same entity or relationship that cannot be simultaneously true
               3. The new fact explicitly updates or supersedes an older fact about the same entity
               4. The new fact implies organizational or structural changes that make previous representations invalid
               5. Facts contain temporal indicators like "now", "previously", "changed to" that signal a state change
               
               Pay special attention to:
               - User ownership: Facts about different users should not contradict each other
               - Domain specificity: Changes in one domain (phones, books, food) should not affect other domains
               - Entity boundaries: Only consider contradictions within the same entity boundary
               
               Do not mark facts as contradictory if they:
               - Provide complementary information about different attributes
               - Represent partial information that can coexist with the new fact
               - Merely add detail without invalidating previous information
               - Belong to different users or entities
               - Represent preferences in different domains
               
               Return the IDs of all existing facts that are contradicted by the new fact. If no contradictions exist, return an empty list.

               <EXISTING FACTS>
               {context['existing_edges']}
               </EXISTING FACTS>

               <NEW FACT>
               {context['new_edge']}
               </NEW FACT>
            """,
        ),
    ]


versions: Versions = {'v1': v1, 'v2': v2}